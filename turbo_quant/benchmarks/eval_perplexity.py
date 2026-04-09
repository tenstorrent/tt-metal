"""Perplexity and KV cache distortion evaluation with TurboQuant.

Two evaluation modes:

1. Model mode: Loads a HuggingFace model, runs forward passes to get baseline
   perplexity, captures real KV cache tensors, and measures TurboQuant distortion
   on those real KV values. Only runs the model ONCE (baseline), then evaluates
   all quantizer variants on the captured KV tensors.

2. Synthetic mode: Simulates attention with random Q/K/V (no model needed).

Usage:
    # With Llama-3.1-8B (BF16, ~16GB RAM):
    python -m turbo_quant.benchmarks.eval_perplexity \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --max-samples 50 --max-seq-len 256

    # Synthetic (no model needed):
    python -m turbo_quant.benchmarks.eval_perplexity --synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import time
import torch
import torch.nn.functional as F

from turbo_quant.quantizer import TurboQuantMSE, TurboQuantProd, OutlierAwareTurboQuant
from turbo_quant.kv_cache import TurboQuantCache


def _make_quantizer(qcfg: dict, head_dim: int, dtype: torch.dtype):
    variant = qcfg["variant"]
    if variant == "outlier":
        return OutlierAwareTurboQuant(
            head_dim=head_dim,
            outlier_bits=qcfg.get("outlier_bits", 3),
            normal_bits=qcfg.get("normal_bits", 2),
            num_outlier_channels=qcfg.get("num_outlier_channels", 32),
            device="cpu",
            dtype=dtype,
        )
    elif variant == "prod":
        return TurboQuantProd(head_dim=head_dim, bits=qcfg.get("bits", 3), device="cpu", dtype=dtype)
    else:
        return TurboQuantMSE(head_dim=head_dim, bits=qcfg.get("bits", 3), device="cpu", dtype=dtype)


def _quantize_dequantize(quantizer, x: torch.Tensor) -> torch.Tensor:
    if isinstance(quantizer, TurboQuantProd):
        return quantizer.dequantize(*quantizer.quantize(x))
    else:
        idx, norms = quantizer.quantize(x)
        return quantizer.dequantize(idx, norms)


def compute_perplexity_with_model(
    model_name: str,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 100,
    max_seq_len: int = 512,
    stride: int = 256,
    quant_configs: list[dict] | None = None,
) -> dict:
    """Run baseline perplexity + KV cache distortion analysis.

    1. Loads model, runs sliding-window perplexity to get baseline PPL
    2. On a subset of windows, captures real KV cache tensors
    3. Measures TurboQuant distortion on those real KV tensors

    Returns dict with baseline_perplexity and per-config KV distortion metrics.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

    print(f"Model: {num_layers} layers, head_dim={head_dim}, kv_heads={num_kv_heads}", flush=True)
    print(f"Loading dataset: {dataset_name}/{dataset_config} ({split})", flush=True)
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate and tokenize
    text = "\n\n".join(t for t in dataset["text"][:max_samples] if t.strip())
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len * max_samples)
    input_ids = encodings.input_ids
    total_input_tokens = input_ids.size(1)
    print(f"Total tokens: {total_input_tokens}", flush=True)

    if quant_configs is None:
        quant_configs = [
            {"name": "mse_2bit", "variant": "mse", "bits": 2},
            {"name": "mse_3bit", "variant": "mse", "bits": 3},
            {"name": "mse_4bit", "variant": "mse", "bits": 4},
            {
                "name": "outlier_2.25bit",
                "variant": "outlier",
                "outlier_bits": 3,
                "normal_bits": 2,
                "num_outlier_channels": 32,
            },
        ]

    # Phase 1: Baseline perplexity + capture KV tensors
    print("\n=== Phase 1: Baseline perplexity (single model pass) ===", flush=True)

    nlls = []
    total_tokens = 0
    captured_kv = []  # List of (keys, values) per window, keys/values are lists over layers
    max_capture_windows = 5  # Only capture KV for a few windows to save memory

    num_windows = max(1, (total_input_tokens - 1) // stride)
    window_idx = 0

    for begin in range(0, total_input_tokens - 1, stride):
        end = min(begin + max_seq_len, total_input_tokens)
        input_chunk = input_ids[:, begin:end]

        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_chunk, output_hidden_states=False, use_cache=True)
            logits = outputs.logits
            past_kv = outputs.past_key_values

        # Perplexity
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, begin + 1 : end].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nlls.append(loss.item())
        total_tokens += shift_labels.numel()

        # Capture KV from this window (detached, to measure quantizer distortion)
        if window_idx < max_capture_windows and past_kv is not None:
            keys_per_layer = []
            values_per_layer = []
            for layer_idx in range(num_layers):
                if hasattr(past_kv, "layers"):
                    # transformers >= 5.x: DynamicCache with .layers[i].keys/.values
                    k = past_kv.layers[layer_idx].keys.detach().clone()
                    v = past_kv.layers[layer_idx].values.detach().clone()
                elif hasattr(past_kv, "key_cache"):
                    # transformers 4.x: DynamicCache with .key_cache[i]/.value_cache[i]
                    k = past_kv.key_cache[layer_idx].detach().clone()
                    v = past_kv.value_cache[layer_idx].detach().clone()
                else:
                    # Legacy tuple format
                    k = past_kv[layer_idx][0].detach().clone()
                    v = past_kv[layer_idx][1].detach().clone()
                keys_per_layer.append(k)
                values_per_layer.append(v)
            captured_kv.append((keys_per_layer, values_per_layer))

        window_idx += 1
        elapsed = time.time() - t0
        running_ppl = math.exp(sum(nlls) / total_tokens) if total_tokens > 0 else float("inf")
        captured_str = f" [KV captured]" if window_idx <= max_capture_windows else ""
        print(f"  window {window_idx}/{num_windows}: {elapsed:.1f}s, ppl={running_ppl:.2f}{captured_str}", flush=True)

        if end >= total_input_tokens:
            break

    baseline_ppl = math.exp(sum(nlls) / total_tokens) if total_tokens > 0 else float("inf")
    print(f"\nBaseline perplexity: {baseline_ppl:.2f} ({total_tokens} tokens)", flush=True)

    # Free model to save memory
    del model, outputs
    import gc

    gc.collect()

    # Phase 2: KV cache distortion analysis
    print(f"\n=== Phase 2: KV cache distortion ({len(captured_kv)} windows, {num_layers} layers) ===", flush=True)

    results = {
        "baseline_perplexity": baseline_ppl,
        "total_tokens": total_tokens,
        "model": model_name,
        "num_layers": num_layers,
        "head_dim": head_dim,
        "kv_distortion": {},
    }

    for qcfg in quant_configs:
        name = qcfg["name"]
        print(f"\n  {name}:", flush=True)

        quantizer = _make_quantizer(qcfg, head_dim, dtype=torch.float32)

        key_mses = []
        val_mses = []
        key_cosines = []
        val_cosines = []

        for win_idx, (keys_list, vals_list) in enumerate(captured_kv):
            for layer_idx in range(num_layers):
                k = keys_list[layer_idx].float()
                v = vals_list[layer_idx].float()

                k_rec = _quantize_dequantize(quantizer, k)
                v_rec = _quantize_dequantize(quantizer, v)

                # MSE
                key_mses.append(((k - k_rec) ** 2).mean().item())
                val_mses.append(((v - v_rec) ** 2).mean().item())

                # Cosine similarity (flattened per-layer)
                key_cosines.append(F.cosine_similarity(k.flatten().unsqueeze(0), k_rec.flatten().unsqueeze(0)).item())
                val_cosines.append(F.cosine_similarity(v.flatten().unsqueeze(0), v_rec.flatten().unsqueeze(0)).item())

        avg_key_mse = sum(key_mses) / len(key_mses)
        avg_val_mse = sum(val_mses) / len(val_mses)
        avg_key_cos = sum(key_cosines) / len(key_cosines)
        avg_val_cos = sum(val_cosines) / len(val_cosines)

        results["kv_distortion"][name] = {
            "key_mse": avg_key_mse,
            "value_mse": avg_val_mse,
            "key_cosine_sim": avg_key_cos,
            "value_cosine_sim": avg_val_cos,
        }

        print(f"    Key   MSE={avg_key_mse:.6f}  cosine={avg_key_cos:.6f}", flush=True)
        print(f"    Value MSE={avg_val_mse:.6f}  cosine={avg_val_cos:.6f}", flush=True)

    return results


def synthetic_perplexity_simulation(
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    seq_lens: list[int] | None = None,
    quant_configs: list[dict] | None = None,
) -> dict:
    """Simulate the effect of KV cache quantization on attention output quality.

    Instead of running a full model, this:
    1. Generates random Q, K, V tensors
    2. Computes ground-truth attention output
    3. Quantizes K, V via TurboQuant, recomputes attention
    4. Measures output MSE and attention score correlation

    This isolates the quantizer's impact on attention computation.
    """
    if seq_lens is None:
        seq_lens = [64, 256, 1024]

    if quant_configs is None:
        quant_configs = [
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
            {
                "name": "outlier_3.5bit",
                "variant": "outlier",
                "outlier_bits": 4,
                "normal_bits": 3,
                "num_outlier_channels": 64,
            },
        ]

    results = {}
    torch.manual_seed(42)

    for seq_len in seq_lens:
        print(f"\n--- seq_len = {seq_len} ---")
        results[seq_len] = {}

        Q = torch.randn(1, num_heads, 1, head_dim)
        K = torch.randn(1, num_heads, seq_len, head_dim)
        V = torch.randn(1, num_heads, seq_len, head_dim)

        scale = head_dim**-0.5
        scores_true = (Q @ K.transpose(-2, -1)) * scale
        attn_true = F.softmax(scores_true, dim=-1)
        out_true = attn_true @ V

        for qcfg in quant_configs:
            name = qcfg["name"]
            quantizer = _make_quantizer(qcfg, head_dim, dtype=torch.float32)
            K_q = _quantize_dequantize(quantizer, K)
            V_q = _quantize_dequantize(quantizer, V)

            scores_q = (Q @ K_q.transpose(-2, -1)) * scale
            attn_q = F.softmax(scores_q, dim=-1)
            out_q = attn_q @ V_q

            output_mse = ((out_true - out_q) ** 2).mean().item()
            score_cosine = F.cosine_similarity(
                scores_true.flatten().unsqueeze(0),
                scores_q.flatten().unsqueeze(0),
            ).item()
            attn_kl = F.kl_div(
                attn_q.log().clamp(min=-100),
                attn_true,
                reduction="batchmean",
                log_target=False,
            ).item()

            results[seq_len][name] = {
                "output_mse": output_mse,
                "score_cosine_sim": score_cosine,
                "attention_kl_div": attn_kl,
            }
            print(f"  {name:25s}  output_MSE={output_mse:.6f}  score_cos={score_cosine:.6f}  attn_KL={attn_kl:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="TurboQuant perplexity evaluation")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic simulation instead of real model")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")

    args = parser.parse_args()

    if args.synthetic:
        results = synthetic_perplexity_simulation()
    else:
        results = compute_perplexity_with_model(
            model_name=args.model,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            max_samples=args.max_samples,
            max_seq_len=args.max_seq_len,
            stride=args.stride,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
