"""HuggingFace-compatible KV cache with TurboQuant compression.

Implements the transformers 5.x Cache protocol:
  - TurboQuantLayer: per-layer compressed storage (replaces DynamicLayer)
  - TurboQuantCache: top-level cache (subclasses Cache)
"""

from __future__ import annotations

import torch
from typing import Optional

from turbo_quant.quantizer import TurboQuantMSE, TurboQuantProd, OutlierAwareTurboQuant
from turbo_quant.bitpack import pack, unpack


def _make_quantizer(
    variant,
    head_dim,
    bits,
    seed,
    device,
    dtype,
    outlier_bits=3,
    normal_bits=2,
    num_outlier_channels=32,
    outlier_mode="static",
):
    if variant == "outlier":
        return OutlierAwareTurboQuant(
            head_dim=head_dim,
            outlier_bits=outlier_bits,
            normal_bits=normal_bits,
            num_outlier_channels=num_outlier_channels,
            outlier_mode=outlier_mode,
            seed=seed,
            device=device,
            dtype=dtype,
        )
    elif variant == "prod":
        return TurboQuantProd(head_dim=head_dim, bits=bits, seed=seed, device=device, dtype=dtype)
    else:
        return TurboQuantMSE(head_dim=head_dim, bits=bits, seed=seed, device=device, dtype=dtype)


class TurboQuantLayer:
    """Per-layer compressed KV storage, compatible with transformers' CacheLayerMixin.

    Stores quantized indices + norms instead of raw key/value tensors.
    Dequantizes on read to return full-precision tensors for attention.
    """

    def __init__(self, quantizer, head_dim: int, bits: int, variant: str, use_bitpack: bool):
        self.quantizer = quantizer
        self.head_dim = head_dim
        self.bits = bits
        self.variant = variant
        self.use_bitpack = use_bitpack

        # Compressed storage
        self._key_compressed: dict | None = None
        self._val_compressed: dict | None = None

        # Dequantized views (for .keys/.values compatibility)
        self._keys_deq: torch.Tensor | None = None
        self._values_deq: torch.Tensor | None = None

        self.is_initialized = False

    @property
    def keys(self) -> torch.Tensor | None:
        if self._key_compressed is None:
            return None
        if self._keys_deq is None:
            self._keys_deq = self._dequantize(self._key_compressed)
        return self._keys_deq

    @keys.setter
    def keys(self, value):
        # Allow setting for compatibility; store raw if needed
        pass

    @property
    def values(self) -> torch.Tensor | None:
        if self._val_compressed is None:
            return None
        if self._values_deq is None:
            self._values_deq = self._dequantize(self._val_compressed)
        return self._values_deq

    @values.setter
    def values(self, value):
        pass

    @property
    def device(self):
        if self._key_compressed is not None:
            return self._key_compressed["indices"].device
        return torch.device("cpu")

    @property
    def dtype(self):
        if self._key_compressed is not None:
            return self._key_compressed["norms"].dtype
        return torch.float32

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Quantize new tokens
        new_key_q = self._quantize(key_states)
        new_val_q = self._quantize(value_states)

        # Append to existing compressed cache
        self._key_compressed = self._append(self._key_compressed, new_key_q)
        self._val_compressed = self._append(self._val_compressed, new_val_q)

        # Invalidate dequantized cache
        self._keys_deq = None
        self._values_deq = None

        # Return full dequantized tensors
        return self.keys, self.values

    def get_seq_length(self) -> int:
        if self._key_compressed is None:
            return 0
        # Sequence dim is always dim=2 (batch, heads, seq, ...)
        return self._key_compressed["norms"].shape[2]

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        kv_length = self.get_seq_length() + query_length
        return kv_length, 0

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self) -> bool:
        return False

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        if self._key_compressed is not None:
            self._key_compressed = {k: v.index_select(0, beam_idx) for k, v in self._key_compressed.items()}
            self._val_compressed = {k: v.index_select(0, beam_idx) for k, v in self._val_compressed.items()}
            self._keys_deq = None
            self._values_deq = None

    def crop(self, max_length: int) -> None:
        if self._key_compressed is not None:
            self._key_compressed = {k: v[:, :, :max_length] for k, v in self._key_compressed.items()}
            self._val_compressed = {k: v[:, :, :max_length] for k, v in self._val_compressed.items()}
            self._keys_deq = None
            self._values_deq = None

    def reset(self) -> None:
        self._key_compressed = None
        self._val_compressed = None
        self._keys_deq = None
        self._values_deq = None
        self.is_initialized = False

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self._key_compressed is not None:
            self._key_compressed = {k: v.repeat_interleave(repeats, dim=0) for k, v in self._key_compressed.items()}
            self._val_compressed = {k: v.repeat_interleave(repeats, dim=0) for k, v in self._val_compressed.items()}
            self._keys_deq = None
            self._values_deq = None

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self._key_compressed is not None:
            self._key_compressed = {k: v.index_select(0, indices) for k, v in self._key_compressed.items()}
            self._val_compressed = {k: v.index_select(0, indices) for k, v in self._val_compressed.items()}
            self._keys_deq = None
            self._values_deq = None

    def offload(self) -> None:
        pass

    def prefetch(self) -> None:
        pass

    # --- Internal ---

    def _quantize(self, x: torch.Tensor) -> dict:
        if self.variant == "prod":
            mse_idx, mse_norms, qjl_signs, res_norms = self.quantizer.quantize(x)
            return {"indices": mse_idx, "norms": mse_norms, "qjl_signs": qjl_signs, "residual_norms": res_norms}
        else:
            indices, norms = self.quantizer.quantize(x)
            if self.use_bitpack:
                indices = pack(indices, self.bits)
            return {"indices": indices, "norms": norms}

    def _dequantize(self, compressed: dict) -> torch.Tensor:
        if self.variant == "prod":
            return self.quantizer.dequantize(
                compressed["indices"],
                compressed["norms"],
                compressed["qjl_signs"],
                compressed["residual_norms"],
            )
        else:
            indices = compressed["indices"]
            if self.use_bitpack:
                indices = unpack(indices, self.bits, self.head_dim)
            return self.quantizer.dequantize(indices, compressed["norms"])

    def _append(self, existing: dict | None, new: dict) -> dict:
        if existing is None:
            return new
        return {k: torch.cat([existing[k], new[k]], dim=2) for k in new}

    def memory_usage_bytes(self) -> int:
        total = 0
        for compressed in [self._key_compressed, self._val_compressed]:
            if compressed is not None:
                for v in compressed.values():
                    total += v.nelement() * v.element_size()
        return total


class TurboQuantCache:
    """Drop-in replacement for HuggingFace's DynamicCache that stores
    key/value tensors in TurboQuant-compressed form.

    Implements the transformers 5.x Cache protocol with a list of
    TurboQuantLayer objects.
    """

    def __init__(
        self,
        num_layers: int = 32,
        head_dim: int = 128,
        bits: int = 3,
        variant: str = "mse",
        seed: int = 42,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
        outlier_bits: int = 3,
        normal_bits: int = 2,
        num_outlier_channels: int = 32,
        outlier_mode: str = "static",
        use_bitpack: bool = True,
    ):
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.bits = bits
        self.variant = variant
        self.device = device
        self.dtype = dtype
        self.use_bitpack = use_bitpack and variant == "mse"

        # Shared quantizer (rotation matrix + codebook are the same across layers)
        self.quantizer = _make_quantizer(
            variant,
            head_dim,
            bits,
            seed,
            device,
            dtype,
            outlier_bits,
            normal_bits,
            num_outlier_channels,
            outlier_mode,
        )

        # Create layers — all share the same quantizer
        self.layers: list[TurboQuantLayer] = [
            TurboQuantLayer(self.quantizer, head_dim, bits, variant, self.use_bitpack) for _ in range(num_layers)
        ]

        self._seen_tokens = 0
        # Used by Cache protocol for lazy layer creation
        self.layer_class_to_replicate = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Extend layers if needed (lazy init for models that don't declare num_layers upfront)
        while len(self.layers) <= layer_idx:
            self.layers.append(
                TurboQuantLayer(self.quantizer, self.head_dim, self.bits, self.variant, self.use_bitpack)
            )

        return self.layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self._seen_tokens = 0

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self.layers:
            layer.batch_select_indices(indices)

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @property
    def is_initialized(self) -> bool:
        return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * len(self.layers)

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def max_batch_size(self) -> int:
        return 0

    @property
    def max_cache_len(self) -> int:
        return 0

    def offload(self, layer_idx: int, only_non_sliding: bool = True) -> None:
        pass

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True) -> None:
        pass

    def early_initialization(self, *args, **kwargs) -> None:
        pass

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        return layer.keys, layer.values

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, None  # None = no sliding window tensor

    def memory_usage_bytes(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].memory_usage_bytes()
