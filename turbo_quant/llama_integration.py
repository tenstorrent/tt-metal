"""Integration hooks for applying TurboQuant KV cache to HuggingFace Llama models."""

from __future__ import annotations

import torch

from turbo_quant.kv_cache import TurboQuantCache


def create_turbo_quant_cache(
    model,
    bits: int = 3,
    variant: str = "mse",
    seed: int = 42,
    outlier_bits: int = 3,
    normal_bits: int = 2,
    num_outlier_channels: int = 32,
    outlier_mode: str = "static",
    use_bitpack: bool = True,
) -> TurboQuantCache:
    """Create a TurboQuantCache sized for a given HuggingFace model.

    Args:
        model: A HuggingFace model (e.g., LlamaForCausalLM).
        bits: Quantization bit-width (for "mse" and "prod" variants).
        variant: "mse", "prod", or "outlier".
        seed: Random seed for quantizer initialization.
        outlier_bits: Bit-width for outlier channels (variant="outlier" only).
        normal_bits: Bit-width for normal channels (variant="outlier" only).
        num_outlier_channels: Number of outlier channels (variant="outlier" only).
        outlier_mode: "static" or "calibration" (variant="outlier" only).
        use_bitpack: Whether to enable bit-packing for memory savings.

    Returns:
        A TurboQuantCache instance configured for the model.
    """
    config = model.config
    num_layers = config.num_hidden_layers
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    return TurboQuantCache(
        num_layers=num_layers,
        head_dim=head_dim,
        bits=bits,
        variant=variant,
        seed=seed,
        device=device,
        dtype=dtype,
        outlier_bits=outlier_bits,
        normal_bits=normal_bits,
        num_outlier_channels=num_outlier_channels,
        outlier_mode=outlier_mode,
        use_bitpack=use_bitpack,
    )


def generate_with_turbo_quant(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    bits: int = 3,
    variant: str = "mse",
    seed: int = 42,
    outlier_bits: int = 3,
    normal_bits: int = 2,
    num_outlier_channels: int = 32,
    outlier_mode: str = "static",
    use_bitpack: bool = True,
    **generate_kwargs,
) -> str:
    """Generate text with TurboQuant-compressed KV cache.

    Args:
        model: HuggingFace LlamaForCausalLM (or compatible).
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        max_new_tokens: Maximum tokens to generate.
        bits: Quantization bit-width (for "mse"/"prod" variants).
        variant: "mse", "prod", or "outlier".
        seed: Random seed.
        outlier_bits: Bit-width for outlier channels (variant="outlier" only).
        normal_bits: Bit-width for normal channels (variant="outlier" only).
        num_outlier_channels: Number of outlier channels (variant="outlier" only).
        outlier_mode: "static" or "calibration" (variant="outlier" only).
        use_bitpack: Whether to enable bit-packing.
        **generate_kwargs: Additional kwargs passed to model.generate().

    Returns:
        Generated text string.
    """
    cache = create_turbo_quant_cache(
        model,
        bits=bits,
        variant=variant,
        seed=seed,
        outlier_bits=outlier_bits,
        normal_bits=normal_bits,
        num_outlier_channels=num_outlier_channels,
        outlier_mode=outlier_mode,
        use_bitpack=use_bitpack,
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            past_key_values=cache,
            **generate_kwargs,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
