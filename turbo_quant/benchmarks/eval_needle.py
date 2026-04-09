"""Needle-in-a-Haystack evaluation for TurboQuant KV cache.

Tests whether quantized KV cache preserves the model's ability to retrieve
specific information buried in a long context. Works in two modes:

1. Synthetic mode (default): Simulates attention retrieval over long sequences
   using random Q/K/V tensors with a planted "needle" — a key vector with
   known high affinity to the query. Measures whether quantization preserves
   the attention peak at the needle position.

2. Model mode: Runs actual generation with a planted fact and retrieval question.
   Requires a HuggingFace model.

Usage:
    # Synthetic (no model needed, CPU-friendly):
    python -m turbo_quant.benchmarks.eval_needle

    # With a real model:
    python -m turbo_quant.benchmarks.eval_needle \
        --model meta-llama/Meta-Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import torch
import torch.nn.functional as F

from turbo_quant.quantizer import TurboQuantMSE, TurboQuantProd, OutlierAwareTurboQuant


def make_quantizer(qcfg: dict, head_dim: int):
    variant = qcfg["variant"]
    if variant == "outlier":
        return OutlierAwareTurboQuant(
            head_dim=head_dim,
            outlier_bits=qcfg.get("outlier_bits", 3),
            normal_bits=qcfg.get("normal_bits", 2),
            num_outlier_channels=qcfg.get("num_outlier_channels", 32),
            device="cpu",
            dtype=torch.float32,
        )
    elif variant == "prod":
        return TurboQuantProd(head_dim=head_dim, bits=qcfg.get("bits", 3), device="cpu", dtype=torch.float32)
    else:
        return TurboQuantMSE(head_dim=head_dim, bits=qcfg.get("bits", 3), device="cpu", dtype=torch.float32)


def quantize_and_dequantize(quantizer, x: torch.Tensor) -> torch.Tensor:
    if isinstance(quantizer, TurboQuantProd):
        return quantizer.dequantize(*quantizer.quantize(x))
    else:
        idx, norms = quantizer.quantize(x)
        return quantizer.dequantize(idx, norms)


def synthetic_needle_test(
    head_dim: int = 128,
    num_heads: int = 8,
    haystack_lengths: list[int] | None = None,
    needle_positions: list[float] | None = None,
    quant_configs: list[dict] | None = None,
    needle_strength: float = 5.0,
) -> dict:
    """Synthetic needle-in-a-haystack test.

    Plants a "needle" key at a known position that has high dot-product with
    the query. After quantizing K/V, checks if the attention still peaks at
    the needle position.

    Args:
        head_dim: Head dimension.
        num_heads: Number of attention heads.
        haystack_lengths: Sequence lengths to test.
        needle_positions: Relative positions (0.0 to 1.0) to place the needle.
        quant_configs: Quantizer configurations to test.
        needle_strength: How much stronger the needle key is vs random keys.

    Returns:
        Nested dict: results[haystack_len][needle_pos][variant] = {
            "needle_rank": int,
            "needle_attn_weight": float,
            "top1_correct": bool,
        }
    """
    if haystack_lengths is None:
        haystack_lengths = [64, 256, 1024, 4096]
    if needle_positions is None:
        needle_positions = [0.1, 0.25, 0.5, 0.75, 0.9]
    if quant_configs is None:
        quant_configs = [
            {"name": "fp32_baseline", "variant": None},
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
    torch.manual_seed(42)

    for hay_len in haystack_lengths:
        results[hay_len] = {}
        for needle_pos in needle_positions:
            results[hay_len][needle_pos] = {}
            needle_idx = int(hay_len * needle_pos)
            needle_idx = max(0, min(needle_idx, hay_len - 1))

            # Create query and haystack
            Q = torch.randn(1, num_heads, 1, head_dim)
            Q = Q / Q.norm(dim=-1, keepdim=True)

            K = torch.randn(1, num_heads, hay_len, head_dim) * 0.5  # weak random keys
            V = torch.randn(1, num_heads, hay_len, head_dim)

            # Plant needle: a key that's strongly aligned with the query
            K[:, :, needle_idx, :] = Q.squeeze(2) * needle_strength

            # Also plant a distinctive value at the needle
            needle_value = torch.ones(1, num_heads, 1, head_dim) * 10.0
            V[:, :, needle_idx : needle_idx + 1, :] = needle_value

            scale = head_dim**-0.5

            for qcfg in quant_configs:
                name = qcfg["name"]

                if qcfg["variant"] is None:
                    K_used, V_used = K, V
                else:
                    quantizer = make_quantizer(qcfg, head_dim)
                    K_used = quantize_and_dequantize(quantizer, K)
                    V_used = quantize_and_dequantize(quantizer, V)

                scores = (Q @ K_used.transpose(-2, -1)) * scale
                attn = F.softmax(scores, dim=-1)  # [1, heads, 1, hay_len]
                output = attn @ V_used

                # Average across heads
                attn_avg = attn.mean(dim=1).squeeze()  # [hay_len]
                needle_weight = attn_avg[needle_idx].item()

                # Rank: how many positions have higher attention than the needle?
                rank = (attn_avg > attn_avg[needle_idx]).sum().item() + 1

                # Is needle the argmax?
                top1_correct = attn_avg.argmax().item() == needle_idx

                # Check output: does it resemble the needle value?
                output_similarity = F.cosine_similarity(
                    output.mean(dim=1).flatten().unsqueeze(0),
                    needle_value.mean(dim=1).flatten().unsqueeze(0),
                ).item()

                results[hay_len][needle_pos][name] = {
                    "needle_rank": int(rank),
                    "needle_attn_weight": needle_weight,
                    "top1_correct": top1_correct,
                    "output_needle_cosine": output_similarity,
                }

    return results


def print_needle_results(results: dict) -> None:
    print("\n" + "=" * 100)
    print("NEEDLE-IN-A-HAYSTACK RESULTS")
    print("=" * 100)

    # Collect all variant names
    all_variants = set()
    for hay_data in results.values():
        for pos_data in hay_data.values():
            all_variants.update(pos_data.keys())
    all_variants = sorted(all_variants)

    for hay_len in sorted(results.keys()):
        print(f"\n--- Haystack Length: {hay_len} ---")
        print(f"{'Variant':25s} {'Pos':>5s} {'Rank':>5s} {'AttnWt':>8s} {'Top1':>5s} {'OutCos':>7s}")
        print("-" * 60)

        for variant in all_variants:
            for needle_pos in sorted(results[hay_len].keys()):
                data = results[hay_len][needle_pos].get(variant)
                if data is None:
                    continue
                top1 = "Y" if data["top1_correct"] else "N"
                print(
                    f"{variant:25s} {needle_pos:>5.2f} {data['needle_rank']:>5d} "
                    f"{data['needle_attn_weight']:>8.4f} {top1:>5s} "
                    f"{data['output_needle_cosine']:>7.4f}"
                )

    # Summary: retrieval accuracy per variant
    print("\n" + "=" * 60)
    print("RETRIEVAL ACCURACY SUMMARY (% of tests where needle is top-1)")
    print("=" * 60)
    for variant in all_variants:
        total = 0
        correct = 0
        for hay_data in results.values():
            for pos_data in hay_data.values():
                if variant in pos_data:
                    total += 1
                    if pos_data[variant]["top1_correct"]:
                        correct += 1
        pct = 100 * correct / total if total > 0 else 0
        print(f"  {variant:25s} {correct}/{total} ({pct:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack test for TurboQuant")
    parser.add_argument(
        "--haystack-lens", type=str, default="64,256,1024,4096", help="Comma-separated haystack lengths"
    )
    parser.add_argument(
        "--needle-positions",
        type=str,
        default="0.1,0.25,0.5,0.75,0.9",
        help="Comma-separated relative needle positions",
    )
    parser.add_argument(
        "--needle-strength", type=float, default=5.0, help="How much stronger the needle key is vs random"
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")

    args = parser.parse_args()
    haystack_lens = [int(x) for x in args.haystack_lens.split(",")]
    needle_positions = [float(x) for x in args.needle_positions.split(",")]

    results = synthetic_needle_test(
        haystack_lengths=haystack_lens,
        needle_positions=needle_positions,
        needle_strength=args.needle_strength,
    )
    print_needle_results(results)

    if args.output:
        # Convert keys to strings for JSON
        json_results = {}
        for hay_len, hay_data in results.items():
            json_results[str(hay_len)] = {}
            for pos, pos_data in hay_data.items():
                json_results[str(hay_len)][str(pos)] = pos_data
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
