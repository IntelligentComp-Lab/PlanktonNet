#!/usr/bin/env python3
"""
Jetson Orin NX Deployment Benchmark

Measures per-model efficiency metrics:
  - Params (M), FLOPs (G), Model Size (MB)
  - Top-1 / Top-5 Accuracy (%)  [when --data-dir provided]
  - Inference Latency FP32 / FP16 (ms ± std, batch=1)
  - Batch Latency (ms/batch) at each throughput batch size
  - Throughput FP32 / FP16 (imgs/s, multiple batch sizes)
  - Peak GPU Memory FP32 / FP16 (MB)
  - Average Power Consumption FP32 / FP16 (mW)
  - Energy Efficiency (imgs/s/W)

Usage:
  cd code/edgeDeployment
  python benchmark_jetson.py --model PlanktonNet \
      --checkpoint ../../model_weight/20251206-225338-PlanktonNet_Le-224/model_best.pth.tar \
      --num-classes 50 --img-size 224 --output results.csv

Accuracy measurement (requires dataset):
  python benchmark_jetson.py --model PlanktonNet \
      --checkpoint .../model_best.pth.tar \
      --data-dir ../../data/ZooplanktonBench50_test --num-classes 50
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
import threading
from collections import OrderedDict

import torch
import torch.nn as nn

# Register PlanktonNet in timm registry before create_model is called
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from PlanktonNet import PlanktonNet  # noqa

from timm.models import create_model, load_checkpoint
from timm.data import create_dataset, create_loader, resolve_data_config

# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------

def _has_flop_counter_mode():
    try:
        from torch.utils.flop_counter import FlopCounterMode  # noqa
        return True
    except ImportError:
        return False


def _has_thop():
    try:
        import thop  # noqa
        return True
    except ImportError:
        return False


def _has_fvcore():
    try:
        import fvcore  # noqa
        return True
    except ImportError:
        return False


def measure_flops(model, input_tensor):
    """
    Measure GFLOPs.
    Priority: torch.utils.flop_counter -> thop -> fvcore -> None
    """
    if _has_flop_counter_mode():
        from torch.utils.flop_counter import FlopCounterMode
        try:
            inp = input_tensor.clone()
            with FlopCounterMode(display=False) as fcm:
                model(inp)
            return round(fcm.get_total_flops() / 1e9, 2)
        except Exception:
            pass

    if _has_thop():
        from thop import profile
        try:
            flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
            return round(flops / 1e9, 2)
        except Exception:
            pass

    if _has_fvcore():
        from fvcore.nn import FlopCountAnalysis
        try:
            flops = FlopCountAnalysis(model, input_tensor).total()
            return round(flops / 1e9, 2)
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Power monitoring (Jetson INA3221 sensors)
# ---------------------------------------------------------------------------

class JetsonPowerMonitor:
    """Samples Jetson INA3221 power sensors in a background thread."""

    def __init__(self):
        self.sensors = self._find_sensors()
        self._samples = []
        self._monitoring = False
        self._thread = None

    @staticmethod
    def _find_sensors():
        sensors = []
        for i2c_dev in glob.glob('/sys/bus/i2c/devices/*'):
            for root, _dirs, files in os.walk(i2c_dev):
                for f in files:
                    if 'power' in f and 'input' in f:
                        sensors.append(os.path.join(root, f))
        return sorted(sensors)

    def read_power_mw(self):
        total_uw = 0
        for s in self.sensors:
            try:
                with open(s) as fh:
                    total_uw += int(fh.read().strip())
            except (OSError, ValueError):
                pass
        return round(total_uw / 1000.0, 2) if total_uw > 0 else None

    def start(self, interval=0.1):
        self._monitoring = True
        self._samples = []
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self._thread.start()

    def _loop(self, interval):
        while self._monitoring:
            p = self.read_power_mw()
            if p is not None:
                self._samples.append(p)
            time.sleep(interval)

    def stop(self):
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._samples:
            return round(sum(self._samples) / len(self._samples), 2), round(max(self._samples), 2)
        return None, None


# ---------------------------------------------------------------------------
# Core measurement routines
# ---------------------------------------------------------------------------

def measure_latency(model, input_tensor, amp_dtype=None, warmup=30, runs=200):
    """Average single-image inference latency (ms) and standard deviation."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)

    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    latencies = []

    with torch.no_grad():
        for _ in range(runs):
            start_ev.record()
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
            end_ev.record()
            torch.cuda.synchronize()
            latencies.append(start_ev.elapsed_time(end_ev))

    avg_ms = sum(latencies) / len(latencies)
    std_ms = (sum((x - avg_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5
    return round(avg_ms, 2), round(std_ms, 2)


def measure_throughput_and_memory(model, input_tensor, batch_size, amp_dtype=None,
                                  warmup=10, runs=50):
    """Returns (throughput_imgs_per_sec, peak_gpu_memory_MB, batch_latency_ms)."""
    batch = input_tensor.repeat(batch_size, 1, 1, 1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = model(batch)
            else:
                _ = model(batch)

    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    with torch.no_grad():
        for _ in range(runs):
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = model(batch)
            else:
                _ = model(batch)
    end_ev.record()
    torch.cuda.synchronize()

    elapsed_s = start_ev.elapsed_time(end_ev) / 1000.0
    throughput = round(batch_size * runs / elapsed_s, 1)
    batch_latency_ms = round(elapsed_s / runs * 1000, 2)
    peak_mem = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

    return throughput, peak_mem, batch_latency_ms


def measure_power_during_inference(model, input_tensor, batch_size, amp_dtype=None,
                                   duration_s=5):
    """Run sustained inference while sampling power; return avg power in mW."""
    pm = JetsonPowerMonitor()
    if not pm.sensors:
        return None, None

    batch = input_tensor.repeat(batch_size, 1, 1, 1)

    model.eval()
    with torch.no_grad():
        for _ in range(10):
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = model(batch)
            else:
                _ = model(batch)
    torch.cuda.synchronize()

    pm.start(interval=0.1)
    t0 = time.time()
    with torch.no_grad():
        while time.time() - t0 < duration_s:
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = model(batch)
            else:
                _ = model(batch)
    torch.cuda.synchronize()
    avg_mw, max_mw = pm.stop()

    idle_mw = _measure_idle_power(pm)
    net_avg = round(avg_mw - idle_mw, 2) if avg_mw is not None and idle_mw is not None else avg_mw

    return avg_mw, net_avg


def _measure_idle_power(pm, duration_s=3):
    time.sleep(0.5)
    pm.start(interval=0.1)
    time.sleep(duration_s)
    avg, _ = pm.stop()
    return avg


# ---------------------------------------------------------------------------
# Accuracy measurement
# ---------------------------------------------------------------------------

def measure_accuracy(model, data_dir, num_classes, img_size, batch_size, device,
                     amp_dtype=None, metrics_avg='macro', workers=4):
    """
    Measure Top-1, Top-5 accuracy and optionally precision/recall/F1.
    Returns dict with accuracy metrics, or empty dict on failure.
    """
    try:
        from timm.utils import accuracy, AverageMeter
    except ImportError:
        print("  [WARN] timm.utils not available, skipping accuracy.")
        return {}

    model.eval()

    # Resolve data config — let model default_cfg supply mean/std/crop_pct
    # (models like tf_efficientnetv2 use (0.5,0.5,0.5) and mobilevit uses (0,0,0)/(1,1,1))
    data_config = resolve_data_config(
        {'input_size': (3, img_size, img_size)},
        model=model,
        verbose=True,
    )
    print(f"  Data config: {data_config}")

    try:
        dataset = create_dataset(root=data_dir, name='', split='validation')
    except Exception:
        try:
            dataset = create_dataset(root=data_dir, name='', split='test')
        except Exception:
            # Try with explicit folder structure (ImageFolder default)
            from torchvision.datasets import ImageFolder
            from torch.utils.data import DataLoader
            try:
                dataset = ImageFolder(data_dir)
            except Exception as e:
                print(f"  [WARN] Cannot load dataset from {data_dir}: {e}")
                return {}

    if 'torchvision' not in str(type(dataset)):
        loader = create_loader(
            dataset,
            input_size=data_config['input_size'],
            batch_size=batch_size,
            use_prefetcher=True,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=workers,
            crop_pct=data_config.get('crop_pct', 1.0),
            pin_memory=True,
            device=device,
        )
        use_timm_loader = True
    else:
        from torch.utils.data import DataLoader
        from torchvision import transforms
        eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        dataset.transform = eval_transform
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=workers, pin_memory=True)
        use_timm_loader = False

    top1_m = AverageMeter()
    top5_m = AverageMeter()

    all_preds = []
    all_targets = []

    if amp_dtype is not None:
        amp_ctx = lambda: torch.amp.autocast('cuda', dtype=amp_dtype)
    else:
        amp_ctx = lambda: torch.no_grad()

    with torch.inference_mode():
        for input, target in loader:
            if use_timm_loader:
                pass  # timm loader already puts data on device
            else:
                input = input.to(device=device)
                target = target.to(device=device)

            with amp_ctx():
                output = model(input)

            batch_size_actual = output.shape[0]
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, min(5, num_classes)))
            top1_m.update(acc1.item(), batch_size_actual)
            top5_m.update(acc5.item(), batch_size_actual)

            predictions = torch.argmax(output, dim=1)
            all_preds.append(predictions.cpu())
            all_targets.append(target.cpu())

    results = OrderedDict([
        ('top1_acc', round(top1_m.avg, 2)),
        ('top5_acc', round(top5_m.avg, 2)),
    ])

    if metrics_avg:
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            all_preds = torch.cat(all_preds).numpy()
            all_targets = torch.cat(all_targets).numpy()
            results['precision'] = round(precision_score(all_targets, all_preds, average=metrics_avg, zero_division=0), 4)
            results['recall'] = round(recall_score(all_targets, all_preds, average=metrics_avg, zero_division=0), 4)
            results['f1_score'] = round(f1_score(all_targets, all_preds, average=metrics_avg, zero_division=0), 4)
        except Exception as e:
            print(f"  [WARN] sklearn metrics failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Main benchmark for one model
# ---------------------------------------------------------------------------

def benchmark_one_model(args):
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    print(f"\n{'='*60}")
    print(f"Benchmark: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    # ---- Build model -------------------------------------------------------
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
    )
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, args.img_size, args.img_size).to(device)

    # ---- Static metrics ----------------------------------------------------
    params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    model_size_mb = round(os.path.getsize(args.checkpoint) / (1024 * 1024), 2)
    flops_g = measure_flops(model, input_tensor)

    print(f"  Params       : {params_m} M")
    print(f"  Model size   : {model_size_mb} MB")
    print(f"  FLOPs        : {flops_g} G" if flops_g else "  FLOPs        : N/A")

    # ---- Accuracy (optional) -----------------------------------------------
    acc_results = {}
    if args.data_dir and os.path.isdir(args.data_dir):
        print(f"\n  --- Accuracy on {args.data_dir} ---")
        acc_results = measure_accuracy(
            model, args.data_dir, args.num_classes, args.img_size,
            args.acc_batch_size, device, amp_dtype=torch.float16,
            metrics_avg=args.metrics_avg, workers=args.workers,
        )
        for k, v in acc_results.items():
            print(f"  {k}: {v}")
    elif args.data_dir:
        print(f"  [WARN] Data dir not found: {args.data_dir}, skipping accuracy.")

    # ---- Latency -----------------------------------------------------------
    lat_fp32, lat_fp32_std = measure_latency(model, input_tensor, amp_dtype=None)
    lat_fp16, lat_fp16_std = measure_latency(model, input_tensor, amp_dtype=torch.float16)
    print(f"  Latency FP32 : {lat_fp32} ± {lat_fp32_std} ms")
    print(f"  Latency FP16 : {lat_fp16} ± {lat_fp16_std} ms")

    # ---- Throughput & Memory at different batch sizes -----------------------
    tp_results = {}
    mem_results = {}
    bl_results = {}  # batch latency

    for bs in args.batch_sizes:
        # FP32
        try:
            tp32, mem32, bl32 = measure_throughput_and_memory(model, input_tensor, bs, amp_dtype=None)
        except RuntimeError as e:
            print(f"  [OOM] FP32 bs={bs}: {e}")
            tp32, mem32, bl32 = None, None, None

        # FP16
        try:
            tp16, mem16, bl16 = measure_throughput_and_memory(model, input_tensor, bs, amp_dtype=torch.float16)
        except RuntimeError as e:
            print(f"  [OOM] FP16 bs={bs}: {e}")
            tp16, mem16, bl16 = None, None, None

        tp_results[(bs, 'fp32')] = tp32
        tp_results[(bs, 'fp16')] = tp16
        mem_results[(bs, 'fp32')] = mem32
        mem_results[(bs, 'fp16')] = mem16
        bl_results[(bs, 'fp32')] = bl32
        bl_results[(bs, 'fp16')] = bl16

        print(f"  Throughput  bs={bs:>2d}  FP32: {tp32 or 'OOM':>10} imgs/s  "
              f"FP16: {tp16 or 'OOM':>10} imgs/s  "
              f"Mem FP32: {mem32 or '-':>6} MB  Mem FP16: {mem16 or '-':>6} MB  "
              f"Lat FP32: {bl32 or '-':>6} ms  Lat FP16: {bl16 or '-':>6} ms")

    # ---- Power consumption -------------------------------------------------
    max_bs = max(args.batch_sizes)
    pow_fp32_avg, pow_fp32_net = measure_power_during_inference(
        model, input_tensor, max_bs, amp_dtype=None)
    pow_fp16_avg, pow_fp16_net = measure_power_during_inference(
        model, input_tensor, max_bs, amp_dtype=torch.float16)

    if pow_fp32_avg is not None:
        print(f"  Power FP32   : {pow_fp32_avg} mW (net: {pow_fp32_net} mW)")
        print(f"  Power FP16   : {pow_fp16_avg} mW (net: {pow_fp16_net} mW)")
    else:
        print("  Power        : sensors not found")

    # ---- Energy efficiency -------------------------------------------------
    imgs_per_watt_fp32 = None
    imgs_per_watt_fp16 = None
    if pow_fp32_net and tp_results.get((max_bs, 'fp32')):
        imgs_per_watt_fp32 = round(tp_results[(max_bs, 'fp32')] / (pow_fp32_net / 1000), 2)
        print(f"  Efficiency FP32: {imgs_per_watt_fp32} imgs/s/W")
    if pow_fp16_net and tp_results.get((max_bs, 'fp16')):
        imgs_per_watt_fp16 = round(tp_results[(max_bs, 'fp16')] / (pow_fp16_net / 1000), 2)
        print(f"  Efficiency FP16: {imgs_per_watt_fp16} imgs/s/W")

    # ---- Build result row --------------------------------------------------
    result = OrderedDict([
        ('model', args.model),
        ('checkpoint', os.path.basename(os.path.dirname(args.checkpoint))),
        ('params_M', params_m),
        ('flops_G', flops_g),
        ('model_size_MB', model_size_mb),
    ])
    # Accuracy
    for k, v in acc_results.items():
        result[k] = v
    # Latency
    result['latency_fp32_ms'] = lat_fp32
    result['latency_fp32_std_ms'] = lat_fp32_std
    result['latency_fp16_ms'] = lat_fp16
    result['latency_fp16_std_ms'] = lat_fp16_std
    # Throughput, memory, batch latency per batch size
    for bs in args.batch_sizes:
        result[f'tp_fp32_bs{bs}'] = tp_results[(bs, 'fp32')]
        result[f'tp_fp16_bs{bs}'] = tp_results[(bs, 'fp16')]
        result[f'lat_fp32_bs{bs}_ms'] = bl_results[(bs, 'fp32')]
        result[f'lat_fp16_bs{bs}_ms'] = bl_results[(bs, 'fp16')]
    # Memory at max batch
    result[f'mem_fp32_bs{max_bs}_MB'] = mem_results[(max_bs, 'fp32')]
    result[f'mem_fp16_bs{max_bs}_MB'] = mem_results[(max_bs, 'fp16')]
    # Power
    result['power_fp32_mW'] = pow_fp32_net
    result['power_fp16_mW'] = pow_fp16_net
    # Energy efficiency
    result['imgs_per_watt_fp32'] = imgs_per_watt_fp32
    result['imgs_per_watt_fp16'] = imgs_per_watt_fp16
    # Hardware / software info
    result['gpu'] = torch.cuda.get_device_name(0)
    result['pytorch_version'] = torch.__version__
    result['cuda_version'] = torch.version.cuda

    # ---- Write CSV ----------------------------------------------------------
    file_exists = os.path.isfile(args.output)
    with open(args.output, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"  -> appended to {args.output}")
    return result


# ---------------------------------------------------------------------------
# Print comparison table from CSV
# ---------------------------------------------------------------------------

def print_table(csv_path):
    """Read the result CSV and print a Markdown comparison table."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data in CSV.")
        return

    headers = [
        'model', 'params_M', 'flops_G', 'model_size_MB',
        'top1_acc', 'top5_acc',
        'latency_fp16_ms', 'latency_fp16_std_ms',
        'tp_fp16_bs1', 'tp_fp16_bs32', 'tp_fp16_bs64',
        'mem_fp16_bs64_MB', 'imgs_per_watt_fp16',
    ]
    available = [h for h in headers if h in rows[0]]

    print("\n## Deployment Benchmark — Jetson Orin NX\n")
    print("| " + " | ".join(available) + " |")
    print("| " + " | ".join(["---"] * len(available)) + " |")
    for row in rows:
        vals = []
        for h in available:
            v = row.get(h, '-')
            if v is None or v == '':
                v = '-'
            vals.append(str(v))
        print("| " + " | ".join(vals) + " |")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Jetson Orin NX Deployment Benchmark')
    parser.add_argument('--model', type=str,
                        help='timm model name (e.g. PlanktonNet, convnextv2_base)')
    parser.add_argument('--checkpoint', type=str,
                        help='path to model_best.pth.tar')
    parser.add_argument('--num-classes', type=int, default=50)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[1, 8, 16, 32, 64])
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to test dataset for accuracy measurement (optional). '
                             'If not provided, accuracy columns will be empty.')
    parser.add_argument('--acc-batch-size', type=int, default=64,
                        help='Batch size for accuracy measurement (default: 64)')
    parser.add_argument('--metrics-avg', type=str, default='macro',
                        choices=['micro', 'macro', 'weighted'],
                        help='Averaging method for precision/recall/F1 (default: macro)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Data loading workers (default: 4)')
    parser.add_argument('--output', type=str, default='benchmark_results.csv')
    parser.add_argument('--summary', type=str, metavar='CSV',
                        help='Print a comparison table from existing CSV and exit.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.summary:
        print_table(args.summary)
        sys.exit(0)

    if not args.model or not args.checkpoint:
        print("ERROR: --model and --checkpoint required for benchmarking.", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script must run on Jetson Orin NX.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {gpu_name}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    benchmark_one_model(args)
