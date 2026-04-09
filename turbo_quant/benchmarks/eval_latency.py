"""Latency and memory benchmarking for TurboQuant KV cache.

Measures quantize/dequantize throughput and compressed memory usage
across different configurations, sequence lengths, and batch sizes.

Usage:
    python -m turbo_quant.benchmarks.eval_latency
    python -m turbo_quant.benchmarks.eval_latency --seq-lens 128,512,2048,8192
"""

from __future__ import annotations

import argparse
import json
import time
import torch

from turbo_quant.quantizer import TurboQuantMSE, TurboQuantProd, OutlierAwareTurboQuant
from turbo_quant.kv_cache import TurboQuantCache


def benchmark_quantizer_throughput(
    head_dim: int = 128,
    num_heads: int = 8,
    seq_lens: list[int] | None = None,
    num_warmup: int = 3,
    num_iters: int = 20,
) -> dict:
    """Benchmark raw quantize/dequantize latency for each quantizer variant.

    Returns:
        Dict keyed by (variant_name, seq_len) → {quantize_ms, dequantize_ms, tokens_per_sec}.
    """
    if seq_lens is None:
        seq_lens = [64, 256, 1024, 4096]

    configs = [
        ("mse_2bit", lambda: TurboQuantMSE(head_dim=head_dim, bits=2, device="cpu", dtype=torch.float32)),
        ("mse_3bit", lambda: TurboQuantMSE(head_dim=head_dim, bits=3, device="cpu", dtype=torch.float32)),
        ("mse_4bit", lambda: TurboQuantMSE(head_dim=head_dim, bits=4, device="cpu", dtype=torch.float32)),
        ("prod_3bit", lambda: TurboQuantProd(head_dim=head_dim, bits=3, device="cpu", dtype=torch.float32)),
        (
            "outlier_2.25bit",
            lambda: OutlierAwareTurboQuant(
                head_dim=head_dim,
                outlier_bits=3,
                normal_bits=2,
                num_outlier_channels=32,
                device="cpu",
                dtype=torch.float32,
            ),
        ),
    ]

    results = {}

    for variant_name, make_quantizer in configs:
        quantizer = make_quantizer()
        is_prod = "prod" in variant_name
        is_outlier = "outlier" in variant_name

        for seq_len in seq_lens:
            x = torch.randn(1, num_heads, seq_len, head_dim)

            # Warmup
            for _ in range(num_warmup):
                if is_prod:
                    packed = quantizer.quantize(x)
                    quantizer.dequantize(*packed)
                else:
                    idx, norms = quantizer.quantize(x)
                    quantizer.dequantize(idx, norms)

            # Benchmark quantize
            t0 = time.perf_counter()
            for _ in range(num_iters):
                if is_prod:
                    packed = quantizer.quantize(x)
                else:
                    idx, norms = quantizer.quantize(x)
            quant_ms = (time.perf_counter() - t0) / num_iters * 1000

            # Benchmark dequantize
            t0 = time.perf_counter()
            for _ in range(num_iters):
                if is_prod:
                    quantizer.dequantize(*packed)
                else:
                    quantizer.dequantize(idx, norms)
            dequant_ms = (time.perf_counter() - t0) / num_iters * 1000

            total_tokens = seq_len * num_heads
            tokens_per_sec = total_tokens / ((quant_ms + dequant_ms) / 1000)

            results[(variant_name, seq_len)] = {
                "quantize_ms": quant_ms,
                "dequantize_ms": dequant_ms,
                "total_ms": quant_ms + dequant_ms,
                "tokens_per_sec": tokens_per_sec,
            }

    return results


def benchmark_memory_usage(
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    seq_lens: list[int] | None = None,
) -> dict:
    """Compare memory usage between FP16/FP32 baseline and TurboQuant variants."""
    if seq_lens is None:
        seq_lens = [256, 1024, 4096, 8192]

    configs = [
        {"name": "mse_2bit", "variant": "mse", "bits": 2},
        {"name": "mse_3bit", "variant": "mse", "bits": 3},
        {"name": "mse_4bit", "variant": "mse", "bits": 4},
        {"name": "prod_3bit", "variant": "prod", "bits": 3},
        {
            "name": "outlier_2.25bit",
            "variant": "outlier",
            "outlier_bits": 3,
            "normal_bits": 2,
            "num_outlier_channels": 32,
        },
    ]

    results = {}

    for seq_len in seq_lens:
        fp16_bytes_per_layer = 2 * seq_len * num_heads * head_dim * 2  # K+V, 2 bytes each
        fp16_total = fp16_bytes_per_layer * num_layers

        results[seq_len] = {
            "fp16_baseline": {
                "per_layer_bytes": fp16_bytes_per_layer,
                "total_bytes": fp16_total,
                "total_mb": fp16_total / (1024 * 1024),
            }
        }

        for qcfg in configs:
            name = qcfg["name"]
            cache = TurboQuantCache(
                num_layers=num_layers,
                head_dim=head_dim,
                bits=qcfg.get("bits", 3),
                variant=qcfg["variant"],
                device="cpu",
                dtype=torch.float32,
                outlier_bits=qcfg.get("outlier_bits", 3),
                normal_bits=qcfg.get("normal_bits", 2),
                num_outlier_channels=qcfg.get("num_outlier_channels", 32),
            )

            # Fill all layers
            for layer_idx in range(num_layers):
                keys = torch.randn(1, num_heads, seq_len, head_dim)
                values = torch.randn(1, num_heads, seq_len, head_dim)
                cache.update(keys, values, layer_idx=layer_idx)

            total_bytes = sum(cache.memory_usage_bytes(i) for i in range(num_layers))
            per_layer = total_bytes / num_layers

            results[seq_len][name] = {
                "per_layer_bytes": int(per_layer),
                "total_bytes": total_bytes,
                "total_mb": total_bytes / (1024 * 1024),
                "compression_ratio": fp16_total / total_bytes if total_bytes > 0 else 0,
            }

    return results


def benchmark_decode_simulation(
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    prefill_len: int = 512,
    decode_steps: int = 128,
) -> dict:
    """Simulate a prefill + decode workload measuring per-step latency.

    This mirrors real inference: prefill N tokens, then decode one at a time.
    """
    configs = [
        {"name": "mse_3bit", "variant": "mse", "bits": 3},
        {
            "name": "outlier_2.25bit",
            "variant": "outlier",
            "outlier_bits": 3,
            "normal_bits": 2,
            "num_outlier_channels": 32,
        },
    ]

    results = {}

    for qcfg in configs:
        name = qcfg["name"]
        cache = TurboQuantCache(
            num_layers=num_layers,
            head_dim=head_dim,
            bits=qcfg.get("bits", 3),
            variant=qcfg["variant"],
            device="cpu",
            dtype=torch.float32,
            outlier_bits=qcfg.get("outlier_bits", 3),
            normal_bits=qcfg.get("normal_bits", 2),
            num_outlier_channels=qcfg.get("num_outlier_channels", 32),
        )

        # Prefill
        prefill_keys = torch.randn(1, num_heads, prefill_len, head_dim)
        prefill_values = torch.randn(1, num_heads, prefill_len, head_dim)

        t0 = time.perf_counter()
        for layer_idx in range(num_layers):
            cache.update(prefill_keys, prefill_values, layer_idx=layer_idx)
        prefill_ms = (time.perf_counter() - t0) * 1000

        # Decode steps (one token at a time per layer)
        decode_times = []
        for step in range(decode_steps):
            new_key = torch.randn(1, num_heads, 1, head_dim)
            new_value = torch.randn(1, num_heads, 1, head_dim)

            t0 = time.perf_counter()
            for layer_idx in range(num_layers):
                cache.update(new_key, new_value, layer_idx=layer_idx)
            decode_times.append((time.perf_counter() - t0) * 1000)

        avg_decode_ms = sum(decode_times) / len(decode_times)
        final_seq_len = cache.get_seq_length(0)
        final_memory_mb = sum(cache.memory_usage_bytes(i) for i in range(num_layers)) / (1024 * 1024)

        results[name] = {
            "prefill_ms": prefill_ms,
            "prefill_tokens": prefill_len,
            "avg_decode_step_ms": avg_decode_ms,
            "min_decode_step_ms": min(decode_times),
            "max_decode_step_ms": max(decode_times),
            "decode_steps": decode_steps,
            "final_seq_len": final_seq_len,
            "final_memory_mb": final_memory_mb,
        }

    return results


def print_throughput_table(results: dict) -> None:
    print("\n" + "=" * 85)
    print("QUANTIZER THROUGHPUT")
    print("=" * 85)
    print(f"{'Variant':25s} {'SeqLen':>7s} {'Quant(ms)':>10s} {'Dequant(ms)':>12s} {'Total(ms)':>10s} {'Tok/sec':>10s}")
    print("-" * 85)
    for (variant, seq_len), data in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        print(
            f"{variant:25s} {seq_len:>7d} {data['quantize_ms']:>10.2f} "
            f"{data['dequantize_ms']:>12.2f} {data['total_ms']:>10.2f} "
            f"{data['tokens_per_sec']:>10.0f}"
        )


def print_memory_table(results: dict) -> None:
    print("\n" + "=" * 80)
    print("MEMORY USAGE (32 layers, 8 KV heads, head_dim=128)")
    print("=" * 80)

    for seq_len in sorted(results.keys()):
        data = results[seq_len]
        fp16 = data["fp16_baseline"]
        print(f"\n  seq_len = {seq_len}")
        print(f"  {'Config':25s} {'Total(MB)':>10s} {'Compression':>12s}")
        print(f"  {'-'*50}")
        print(f"  {'fp16_baseline':25s} {fp16['total_mb']:>10.2f} {'1.0x':>12s}")
        for name in sorted(k for k in data if k != "fp16_baseline"):
            d = data[name]
            print(f"  {name:25s} {d['total_mb']:>10.2f} {d['compression_ratio']:>11.1f}x")


def print_decode_table(results: dict) -> None:
    print("\n" + "=" * 80)
    print("DECODE SIMULATION (512 prefill + 128 decode steps)")
    print("=" * 80)
    print(f"{'Variant':25s} {'Prefill(ms)':>12s} {'Decode/step(ms)':>16s} {'Memory(MB)':>11s}")
    print("-" * 70)
    for name, data in results.items():
        print(
            f"{name:25s} {data['prefill_ms']:>12.1f} "
            f"{data['avg_decode_step_ms']:>16.2f} "
            f"{data['final_memory_mb']:>11.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="TurboQuant latency and memory benchmarks")
    parser.add_argument("--seq-lens", type=str, default="64,256,1024,4096", help="Comma-separated sequence lengths")
    parser.add_argument("--num-iters", type=int, default=20, help="Iterations per measurement")
    parser.add_argument("--skip-decode", action="store_true", help="Skip decode simulation")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")

    args = parser.parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("Running throughput benchmark...")
    throughput = benchmark_quantizer_throughput(seq_lens=seq_lens, num_iters=args.num_iters)
    print_throughput_table(throughput)

    print("\nRunning memory benchmark...")
    memory = benchmark_memory_usage(seq_lens=seq_lens)
    print_memory_table(memory)

    decode = None
    if not args.skip_decode:
        print("\nRunning decode simulation...")
        decode = benchmark_decode_simulation()
        print_decode_table(decode)

    if args.output:
        # Convert tuple keys to strings for JSON serialization
        throughput_json = {f"{k[0]}_seq{k[1]}": v for k, v in throughput.items()}
        out = {
            "throughput": throughput_json,
            "memory": {str(k): v for k, v in memory.items()},
        }
        if decode:
            out["decode_simulation"] = decode
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
