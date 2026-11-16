#!/usr/bin/env python3
"""
统一 OSR 评估脚本 — 为 plot_osr_figures.py 生成数据
=====================================================
评估 OpenMax 和 PMAL 在各 backbone 上的 OSR 表现。
输出 all_results.json 供绘图脚本使用。

用法 (JSON config 方式):
    python paper/eval_all_methods.py \
        --data-dir ./ --dataset ZooplanktonBench50_padding \
        --configs paper/eval_configs.json \
        --output-dir paper/results \
        --device cuda:0

用法 (CLI 方式, 单方法):
    python paper/eval_all_methods.py \
        --data-dir ./ --dataset ZooplanktonBench50_padding \
        --methods openmax \
        --om-model plankton_net --om-checkpoint path/to/ckpt.pth.tar \
        --output-dir paper/results --device cuda:0
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse, logging, sys, json
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.utils import setup_default_logging
import timm

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from osr_metrics import (
    compute_standard_metrics, compute_paper_metrics,
    search_best_threshold, compute_tau_curve,
)

_logger = logging.getLogger('eval_all')


class BackboneWithProjection(nn.Module):
    def __init__(self, backbone, in_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2: feat = feat.flatten(1)
        return self.projector(feat)


def detect_backbone_dim(backbone, img_size):
    with torch.no_grad():
        dummy = torch.zeros(2, 3, img_size, img_size)
        feat = backbone(dummy)
        if feat.dim() > 2: feat = feat.flatten(1)
        return feat.shape[1]


def create_backbone(model_name, img_size=224, device='cpu'):
    if model_name == 'plankton_net':
        from PlanktonNet import PlanktonNet
        return PlanktonNet(pretrained=False)
    kwargs = dict(pretrained=False, num_classes=0)
    return timm.create_model(model_name, **kwargs)


def get_class_map(dataset):
    for attr in ['class_to_idx', 'parser', 'reader']:
        obj = getattr(dataset, attr, None)
        if obj is not None:
            cm = getattr(obj, 'class_to_idx', None) if attr != 'class_to_idx' else obj
            if cm is not None: return cm
    return None


def make_loader(root, split, img_size, batch_size, workers, device,
                class_map=None, is_training=False, data_config=None):
    if data_config is None:
        data_config = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
                       'interpolation': 'bicubic'}
    path = os.path.join(root, split)
    ds = create_dataset(name='ImageFolder', root=path, is_training=is_training,
                       class_map=class_map, batch_size=batch_size)
    loader = create_loader(ds, (3, img_size, img_size), batch_size,
                          is_training=is_training, mean=data_config['mean'],
                          std=data_config['std'],
                          interpolation=data_config.get('interpolation', 'bicubic'),
                          num_workers=workers, device=device,
                          persistent_workers=workers > 0)
    return loader, ds


class CACModelWrapper(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        from cacUtils import CACModel as _CACModel
        self.model = _CACModel(backbone, feature_dim=feature_dim, num_classes=num_classes)

    def forward(self, x): return self.model(x)

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        sd = ckpt.get('state_dict', ckpt)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        return self.model.load_state_dict(sd, strict=False)


# ── OpenMax Evaluation ──

def eval_openmax(args, device):
    model_name = args.om_model
    checkpoint_path = args.om_checkpoint
    num_classes = args.num_classes
    img_size = args.img_size

    _logger.info(f"[OpenMax] model={model_name}, ckpt={checkpoint_path}")
    if model_name == 'plankton_net':
        from PlanktonNet import PlanktonNet
        model = PlanktonNet(num_classes=num_classes, pretrained=False).to(device)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(device)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    if model_name == 'plankton_net':
        _orig_forward = model.forward
        model.forward = lambda x: model.head(model.forward_features(x))

    data_config = resolve_data_config({'img_size': img_size}, model=model)
    _logger.info(f"[OpenMax] Data config: {data_config}")

    root = os.path.join(args.data_dir, args.dataset)
    loader_train, ds_train = make_loader(root, args.train_split, img_size, args.batch_size, args.workers, device, data_config=data_config)
    class_map = get_class_map(ds_train)
    loader_known, _ = make_loader(root, args.test_known_split, img_size, args.batch_size, args.workers, device, class_map, data_config=data_config)
    loader_unknown, _ = make_loader(root, args.test_unknown_split, img_size, args.batch_size, args.workers, device, data_config=data_config)
    loader_val, _ = make_loader(root, args.val_split, img_size, args.batch_size, args.workers, device, class_map, data_config=data_config)
    loader_val_unknown, _ = make_loader(root, args.val_unknown_split, img_size, args.batch_size, args.workers, device, data_config=data_config)

    from pytorch_ood.detector import OpenMax as oodOpenMax
    _logger.info("[OpenMax] Extracting training logits for Weibull fitting...")
    logits_train, labels_train = [], []
    with torch.no_grad():
        for inp, tgt in tqdm(loader_train, desc="Train logits"):
            logits_train.append(model(inp.to(device)).cpu())
            labels_train.append(tgt.cpu())
    logits_train = torch.cat(logits_train); labels_train = torch.cat(labels_train)

    om = oodOpenMax(None, tailsize=args.om_tailsize, alpha=args.om_alpha)
    om.fit_features(logits_train, labels_train)

    def openmax_scores(loader, desc=""):
        logits_list, tgts_list = [], []
        with torch.no_grad():
            for inp, tgt in tqdm(loader, desc=desc):
                logits_list.append(model(inp.to(device)).cpu())
                tgts_list.append(tgt.cpu())
        logits = torch.cat(logits_list); tgts = torch.cat(tgts_list)
        probs_np = om._openmax.predict(logits.numpy())
        probs = torch.from_numpy(probs_np).float()
        p_unknown = probs[:, 0]
        known_probs, preds = probs[:, 1:].max(dim=1)
        return p_unknown, preds, tgts

    s_k, p_k, t_k = openmax_scores(loader_known, "Known")
    s_u, _, _ = openmax_scores(loader_unknown, "Unknown")
    s_v, p_v, t_v = openmax_scores(loader_val, "Validation")
    s_vu, _, _ = openmax_scores(loader_val_unknown, "Val-Unknown")

    # τ selected entirely on validation set (max F1-score)
    best_th, _ = search_best_threshold(s_v.numpy(), p_v.numpy(), t_v.numpy(), s_vu.numpy())
    # Compute final paper metrics on test set using the fixed τ
    paper_metrics = compute_paper_metrics(
        torch.as_tensor(s_k.numpy()), torch.as_tensor(p_k.numpy()),
        torch.as_tensor(t_k.numpy()), torch.as_tensor(s_u.numpy()), best_th)
    tau_curve = compute_tau_curve(s_v.numpy(), p_v.numpy(), t_v.numpy(), s_vu.numpy())
    std_metrics = compute_standard_metrics(s_k.numpy(), s_u.numpy())

    return dict(
        method='OpenMax', backbone=model_name, checkpoint=checkpoint_path,
        best_threshold=float(best_th),
        paper_metrics=dict(zip(
            ['known_closed', 'known_th', 'unknown_th', 'os_acc', 'f_score'],
            [float(m) for m in paper_metrics])),
        standard_metrics=std_metrics,
        tau_curve={k: v.tolist() if hasattr(v, 'tolist') else list(v) for k, v in tau_curve.items()},
        scores=dict(known=s_k.numpy().tolist(), unknown=s_u.numpy().tolist(),
                   val=s_v.numpy().tolist(), val_unknown=s_vu.numpy().tolist()),
    )


# ── PMAL Evaluation ──

def eval_pmal(args, device):
    model_name = args.pmal_model
    checkpoint_path = args.pmal_checkpoint
    feature_dim = args.pmal_feature_dim
    num_classes = args.num_classes
    img_size = args.img_size

    _logger.info(f"[PMAL] model={model_name}, ckpt={checkpoint_path}")
    from pmal.pmalUtils import PMALModel, PMALoss, PMALWrapper, create_pmal_backbone

    backbone, backbone_dim = create_pmal_backbone(model_name, feature_dim, img_size=img_size)
    model_core = PMALModel(backbone, feature_dim=feature_dim, num_classes=num_classes)
    loss_fn = PMALoss(num_classes, device=device)
    model = PMALWrapper(model_core, loss_fn).to(device)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    limits = model.loss_fn.limits.data if hasattr(model.loss_fn, 'limits') else torch.ones(num_classes).to(device)

    root = os.path.join(args.data_dir, args.dataset)
    loader_train, ds_train = make_loader(root, args.train_split, img_size, args.batch_size, args.workers, device)
    class_map = get_class_map(ds_train)
    loader_known, _ = make_loader(root, args.test_known_split, img_size, args.batch_size, args.workers, device, class_map)
    loader_unknown, _ = make_loader(root, args.test_unknown_split, img_size, args.batch_size, args.workers, device)
    loader_val, _ = make_loader(root, args.val_split, img_size, args.batch_size, args.workers, device, class_map)
    loader_val_unknown, _ = make_loader(root, args.val_unknown_split, img_size, args.batch_size, args.workers, device)

    def pmal_scores(loader, desc=""):
        scores_list, preds_list, tgts_list = [], [], []
        with torch.no_grad():
            for inp, tgt in tqdm(loader, desc=desc):
                inp = inp.to(device)
                logits = model(inp)
                normalized = logits / limits.unsqueeze(0)
                max_logits, _ = normalized.max(dim=1)
                batch_scores = -1.0 * max_logits
                preds = logits.argmax(dim=1)
                scores_list.append(batch_scores.cpu())
                preds_list.append(preds.cpu())
                tgts_list.append(tgt.cpu())
        return torch.cat(scores_list), torch.cat(preds_list), torch.cat(tgts_list)

    s_k, p_k, t_k = pmal_scores(loader_known, "Known")
    s_u, _, _ = pmal_scores(loader_unknown, "Unknown")
    s_v, p_v, t_v = pmal_scores(loader_val, "Validation")
    s_vu, _, _ = pmal_scores(loader_val_unknown, "Val-Unknown")

    # τ selected entirely on validation set (max F1-score)
    best_th, _ = search_best_threshold(s_v.numpy(), p_v.numpy(), t_v.numpy(), s_vu.numpy())
    # Compute final paper metrics on test set using the fixed τ
    paper_metrics = compute_paper_metrics(
        torch.as_tensor(s_k.numpy()), torch.as_tensor(p_k.numpy()),
        torch.as_tensor(t_k.numpy()), torch.as_tensor(s_u.numpy()), best_th)
    std_metrics = compute_standard_metrics(s_k.numpy(), s_u.numpy())
    tau_curve = compute_tau_curve(s_v.numpy(), p_v.numpy(), t_v.numpy(), s_vu.numpy())

    return dict(
        method='PMAL', backbone=model_name, checkpoint=checkpoint_path,
        best_threshold=float(best_th),
        paper_metrics=dict(zip(
            ['known_closed', 'known_th', 'unknown_th', 'os_acc', 'f_score'],
            [float(m) for m in paper_metrics])),
        standard_metrics=std_metrics,
        tau_curve={k: v.tolist() if hasattr(v, 'tolist') else v for k, v in tau_curve.items()},
        scores=dict(known=s_k.numpy().tolist(), unknown=s_u.numpy().tolist(),
                   val=s_v.numpy().tolist(), val_unknown=s_vu.numpy().tolist()),
    )


METHOD_REGISTRY = {'openmax': eval_openmax, 'pmal': eval_pmal}
METHOD_PREFIXES = {'openmax': 'om', 'pmal': 'pmal'}


def parse_args():
    p = argparse.ArgumentParser(description='Unified OSR Evaluation')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--dataset', default='ZooplanktonBench50_padding')
    p.add_argument('--output-dir', default='./paper/results')
    p.add_argument('--train-split', default='train')
    p.add_argument('--val-split', default='validation')
    p.add_argument('--val-unknown-split', default='validation_unknow')
    p.add_argument('--test-known-split', default='test_know')
    p.add_argument('--test-unknown-split', default='test_unknow')
    p.add_argument('--num-classes', type=int, default=50)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--methods', nargs='+', choices=METHOD_REGISTRY, default=list(METHOD_REGISTRY))
    # OpenMax
    p.add_argument('--om-model', default='plankton_net')
    p.add_argument('--om-checkpoint', default=None)
    p.add_argument('--om-tailsize', type=int, default=20)
    p.add_argument('--om-alpha', type=int, default=10)
    # PMAL
    p.add_argument('--pmal-model', default='plankton_net')
    p.add_argument('--pmal-checkpoint', default=None)
    p.add_argument('--pmal-feature-dim', type=int, default=448)
    # Multi-config
    p.add_argument('--configs', default=None, help='JSON config file')
    return p.parse_args()


def build_task_list(args):
    tasks = []
    if args.configs:
        with open(args.configs, 'r') as f:
            config_list = json.load(f)
        for cfg in config_list:
            if cfg['method'] not in METHOD_REGISTRY:
                raise ValueError(f"Unsupported method: {cfg['method']}")
            tasks.append(cfg)
        return tasks

    for method_key in args.methods:
        prefix = METHOD_PREFIXES[method_key]
        cfg = dict(method=method_key,
                   label=f"{method_key}_{getattr(args, f'{prefix}_model')}",
                   model=getattr(args, f'{prefix}_model'),
                   checkpoint=getattr(args, f'{prefix}_checkpoint'))
        if method_key == 'openmax':
            cfg['tailsize'] = args.om_tailsize; cfg['alpha'] = args.om_alpha
        elif method_key == 'pmal': cfg['feature_dim'] = args.pmal_feature_dim
        tasks.append(cfg)
    return tasks


def main():
    setup_default_logging()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = build_task_list(args)
    _logger.info(f"Running {len(tasks)} evaluation(s)")

    all_results = {}
    for i, task in enumerate(tasks):
        method_key = task['method']
        label = task.get('label', f"{method_key}_{i}")
        _logger.info(f"{'='*60}\n[{i+1}/{len(tasks)}] {label}\n{'='*60}")

        task_args = argparse.Namespace(**vars(args))
        prefix = METHOD_PREFIXES[method_key]
        setattr(task_args, f'{prefix}_model', task['model'])
        setattr(task_args, f'{prefix}_checkpoint', task['checkpoint'])
        if method_key == 'openmax':
            setattr(task_args, 'om_tailsize', task.get('tailsize', args.om_tailsize))
            setattr(task_args, 'om_alpha', task.get('alpha', args.om_alpha))
        elif method_key == 'pmal':
            setattr(task_args, 'pmal_feature_dim', task.get('feature_dim', args.pmal_feature_dim))

        try:
            result = METHOD_REGISTRY[method_key](task_args, device)
            all_results[label] = result
            _print_results(result)
        except Exception as e:
            _logger.error(f"Failed to evaluate {label}: {e}", exc_info=True)
            all_results[label] = dict(error=str(e))

    json_path = os.path.join(args.output_dir, 'all_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    _logger.info(f"Results saved to {json_path}")

    for label, res in all_results.items():
        if 'scores' in res:
            npz_path = os.path.join(args.output_dir, f'{label}_scores.npz')
            np.savez(npz_path, known=np.array(res['scores']['known']),
                     unknown=np.array(res['scores']['unknown']),
                     val=np.array(res['scores'].get('val', [])))
            _logger.info(f"Scores saved to {npz_path}")

    _print_summary(all_results)


def _print_results(result):
    if 'error' in result:
        print(f"  ERROR: {result['error']}"); return
    pm = result['paper_metrics']; sm = result['standard_metrics']
    print(f"\n  Method: {result['method']} ({result['backbone']})")
    print(f"  Known(Closed)={pm['known_closed']*100:.2f}% Known(Th)={pm['known_th']*100:.2f}%")
    print(f"  Unknown(Th)={pm['unknown_th']*100:.2f}% OS-Acc={pm['os_acc']*100:.2f}% F1={pm['f_score']*100:.2f}%")
    print(f"  AUROC={sm['auroc']*100:.2f}% AUPR={sm['aupr']*100:.2f}% FPR95={sm['fpr95']*100:.2f}%")


def _print_summary(all_results):
    print("\n" + "=" * 90)
    print(f"  {'Method':<12} {'Backbone':<18} {'AUROC':>8} {'AUPR':>8} {'FPR95':>8} {'F1':>8} {'Unk(Th)':>8}")
    print("-" * 90)
    for key, res in all_results.items():
        if 'error' in res: continue
        pm = res['paper_metrics']; sm = res['standard_metrics']
        print(f"  {res['method']:<12} {res['backbone']:<18} "
              f"{sm['auroc']*100:>7.2f}% {sm['aupr']*100:>7.2f}% "
              f"{sm['fpr95']*100:>7.2f}% {pm['f_score']*100:>7.2f}% {pm['unknown_th']*100:>7.2f}%")
    print("=" * 90)


if __name__ == '__main__':
    main()
