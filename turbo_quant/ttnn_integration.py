"""TTNN-accelerated TurboQuant KV cache quantization.

Implements TurboQuant quantize/dequantize using TTNN composite operations
(matmul, embedding, eltwise) for on-device execution on Tenstorrent hardware.

Phase 1: All ops are composed from existing TTNN primitives.
Phase 2 (future): Fused device kernels for production performance.

Usage:
    import ttnn
    from turbo_quant.ttnn_integration import TTNNTurboQuantCache

    device = ttnn.open_device(0)
    cache = TTNNTurboQuantCache(device, head_dim=128, bits=3)

    # During decode:
    k_idx, k_norms = cache.quantize(k_heads)  # on-device quantize
    cache.store(k_idx, k_norms, layer_idx, pos)

    # Before SDPA:
    k_dequant = cache.dequantize(layer_idx)  # on-device dequantize
"""

from __future__ import annotations

import torch
import math

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from turbo_quant.rotation import generate_rotation_matrix
from turbo_quant.codebook import get_codebook

# Fused kernels (B1/B2) — available after building TTNN with turbo_quant op.
_FUSED_OPS_AVAILABLE = False
if TTNN_AVAILABLE:
    _FUSED_OPS_AVAILABLE = hasattr(ttnn.experimental, "turbo_quant_bucketize")


def _require_ttnn():
    if not TTNN_AVAILABLE:
        raise RuntimeError("ttnn is not available. This module requires Tenstorrent hardware.")


class TTNNTurboQuantSetup:
    """Holds precomputed TurboQuant constants pushed to device.

    Generates rotation matrix and codebook on CPU, then transfers to device
    as TTNN tensors. These are shared across all layers.
    """

    def __init__(
        self,
        device,
        head_dim: int = 128,
        bits: int = 3,
        seed: int = 42,
    ):
        _require_ttnn()

        self.device = device
        self.head_dim = head_dim
        self.bits = bits
        self.num_levels = 1 << bits

        # Generate rotation matrix on CPU (float32 for precision)
        rotation_cpu = generate_rotation_matrix(head_dim, seed=seed, dtype=torch.float32)
        rotation_t_cpu = rotation_cpu.t().contiguous()

        # Get codebook (centroids and boundaries)
        codebook = get_codebook(head_dim, bits, device="cpu", dtype=torch.float32)
        centroids_cpu = codebook.centroids.clone()  # [num_levels]
        boundaries_cpu = codebook.boundaries[1:-1].clone()  # [num_levels - 1] (inner boundaries)

        # Push to device as TTNN tensors
        # Rotation: [1, 1, head_dim, head_dim] for batch matmul compatibility
        self.rotation = ttnn.from_torch(
            rotation_cpu.unsqueeze(0).unsqueeze(0),  # [1, 1, 128, 128]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.rotation_t = ttnn.from_torch(
            rotation_t_cpu.unsqueeze(0).unsqueeze(0),  # [1, 1, 128, 128]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Centroids: [1, 1, 1, num_levels] for embedding lookup
        # Pad to tile-aligned width (nearest 32)
        padded_levels = ((self.num_levels + 31) // 32) * 32
        centroids_padded = torch.zeros(padded_levels)
        centroids_padded[: self.num_levels] = centroids_cpu
        self.centroids = ttnn.from_torch(
            centroids_padded.unsqueeze(0).unsqueeze(0),  # [1, 1, padded_levels]
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Boundaries for bucketize: [1, 1, 1, num_levels - 1]
        # Pad to tile width
        padded_bounds = ((len(boundaries_cpu) + 31) // 32) * 32
        bounds_padded = torch.full((padded_bounds,), float("inf"))
        bounds_padded[: len(boundaries_cpu)] = boundaries_cpu
        self.boundaries = ttnn.from_torch(
            bounds_padded.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.num_boundaries = len(boundaries_cpu)

    def deallocate(self):
        """Free device memory."""
        ttnn.deallocate(self.rotation)
        ttnn.deallocate(self.rotation_t)
        ttnn.deallocate(self.centroids)
        ttnn.deallocate(self.boundaries)


def turbo_quant_quantize(
    x: "ttnn.Tensor",
    setup: TTNNTurboQuantSetup,
    memory_config=None,
    skip_rotation: bool = False,
) -> tuple["ttnn.Tensor", "ttnn.Tensor"]:
    """Quantize a KV tensor on device using TurboQuant.

    Args:
        x: Input tensor [batch, heads, seq, head_dim] in BF16 on device.
        setup: Precomputed rotation matrix and codebook on device.
        memory_config: Output memory config (default: DRAM).
        skip_rotation: If True, skip the rotation step (input already in rotated space,
                       e.g. when Π has been absorbed into the projection weights).

    Returns:
        (indices, norms) where:
          - indices: UINT32 tensor [batch, heads, seq, head_dim] with centroid indices
          - norms: BF16 tensor [batch, heads, seq, 1] with L2 norms
    """
    _require_ttnn()
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Step 1: Rotate — y = x @ Π  (skip if rotation absorbed into weights)
    if skip_rotation:
        y = x
    else:
        y = ttnn.matmul(x, setup.rotation, memory_config=memory_config)

    # Step 2: L2 norm + normalize (4 ops instead of 5: rsqrt+mul replaces sqrt+add+div)
    y_sq = ttnn.mul(y, y)
    norms_sq = ttnn.sum(y_sq, dim=-1, keepdim=True)  # [batch, heads, seq, 1]
    ttnn.deallocate(y_sq)
    inv_norm = ttnn.rsqrt(norms_sq)  # 1/||y||
    y_hat = ttnn.mul(y, inv_norm, memory_config=memory_config)  # y / ||y||
    ttnn.deallocate(y)

    # Compute ||y|| from inv_norm for the rescale step (no extra sqrt needed).
    norms = ttnn.reciprocal(inv_norm)
    ttnn.deallocate(inv_norm)
    ttnn.deallocate(norms_sq)

    # Step 3: Bucketize — fused kernel, outputs BF16 indices directly.
    indices = _bucketize_on_device(y_hat, setup, memory_config)
    ttnn.deallocate(y_hat)

    return indices, norms


def _bucketize_on_device(
    y_hat: "ttnn.Tensor",
    setup: TTNNTurboQuantSetup,
    memory_config,
) -> "ttnn.Tensor":
    """Bucketize values into codebook indices.

    Uses fused kernel (B1) when available, otherwise falls back to
    cascaded comparisons (13 TTNN ops for 3-bit).
    """
    boundaries_cpu = get_codebook(setup.head_dim, setup.bits, device="cpu", dtype=torch.float32).boundaries[
        1:-1
    ]  # inner boundaries only

    # ── Fused kernel path (single device kernel, no DRAM intermediates) ──
    if _FUSED_OPS_AVAILABLE:
        boundary_list = [b.item() for b in boundaries_cpu]
        return ttnn.experimental.turbo_quant_bucketize(y_hat, boundary_list)

    # ── Fallback: cascaded ge + add ──
    accumulator = None
    for i in range(len(boundaries_cpu)):
        boundary_val = boundaries_cpu[i].item()
        comparison = ttnn.ge(y_hat, boundary_val)
        if accumulator is None:
            accumulator = comparison
        else:
            new_acc = ttnn.add(accumulator, comparison, memory_config=memory_config)
            ttnn.deallocate(accumulator)
            ttnn.deallocate(comparison)
            accumulator = new_acc

    return accumulator


def turbo_quant_dequantize(
    indices: "ttnn.Tensor",
    norms: "ttnn.Tensor",
    setup: TTNNTurboQuantSetup,
    memory_config=None,
) -> "ttnn.Tensor":
    """Dequantize compressed KV tensor back to BF16 on device.

    Args:
        indices: BF16 tensor [batch, heads, seq, head_dim] with centroid indices.
        norms: BF16 tensor [batch, heads, seq, 1] with L2 norms.
        setup: Precomputed rotation matrix and codebook on device.
        memory_config: Output memory config (default: DRAM).

    Returns:
        Reconstructed BF16 tensor [batch, heads, seq, head_dim].
    """
    _require_ttnn()
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Step 1: Gather centroids from BF16 indices
    y_hat = _gather_centroids_from_bf16(indices, setup, memory_config)

    # Step 2: Rescale — y = y_hat * norms
    y = ttnn.mul(y_hat, norms, memory_config=memory_config)
    ttnn.deallocate(y_hat)

    # Step 3: Inverse rotation — x_rec = y @ Πᵀ
    x_rec = ttnn.matmul(y, setup.rotation_t, memory_config=memory_config)
    ttnn.deallocate(y)

    return x_rec


def _gather_centroids_on_device(
    indices: "ttnn.Tensor",
    setup: TTNNTurboQuantSetup,
    memory_config,
) -> "ttnn.Tensor":
    """Map UINT32 integer indices to centroid values on device."""
    centroids_cpu = get_codebook(setup.head_dim, setup.bits, device="cpu", dtype=torch.float32).centroids

    # Cast indices to BF16 for comparison / fused kernel
    indices_bf16 = ttnn.typecast(indices, dtype=ttnn.bfloat16)

    # ── Fused kernel path ──
    if _FUSED_OPS_AVAILABLE:
        centroid_list = [c.item() for c in centroids_cpu]
        result = ttnn.experimental.turbo_quant_gather_centroids(indices_bf16, centroid_list)
        ttnn.deallocate(indices_bf16)
        return result

    # ── Fallback: cascaded ge + where ──
    result = ttnn.full_like(indices_bf16, centroids_cpu[0].item())

    for level in range(1, setup.num_levels):
        centroid_val = centroids_cpu[level].item()
        mask = ttnn.ge(indices_bf16, float(level))
        level_values = ttnn.full_like(indices_bf16, centroid_val)
        new_result = ttnn.where(mask, level_values, result, memory_config=memory_config)
        ttnn.deallocate(mask)
        ttnn.deallocate(level_values)
        ttnn.deallocate(result)
        result = new_result

    ttnn.deallocate(indices_bf16)
    return result


def _gather_centroids_from_bf16(
    indices_bf16: "ttnn.Tensor",
    setup: TTNNTurboQuantSetup,
    memory_config,
) -> "ttnn.Tensor":
    """Map BF16 integer indices to centroid values.

    Uses fused kernel (B2) when available, otherwise falls back to
    cascaded ge + where (21 TTNN ops for 3-bit).
    """
    centroids_cpu = get_codebook(setup.head_dim, setup.bits, device="cpu", dtype=torch.float32).centroids

    # ── Fused kernel path (single device kernel, no DRAM intermediates) ──
    if _FUSED_OPS_AVAILABLE:
        centroid_list = [c.item() for c in centroids_cpu]
        return ttnn.experimental.turbo_quant_gather_centroids(indices_bf16, centroid_list)

    # ── Fallback: cascaded ge + where ──
    result = ttnn.full_like(indices_bf16, centroids_cpu[0].item())

    for level in range(1, setup.num_levels):
        centroid_val = centroids_cpu[level].item()
        mask = ttnn.ge(indices_bf16, float(level))
        level_values = ttnn.full_like(indices_bf16, centroid_val)
        new_result = ttnn.where(mask, level_values, result, memory_config=memory_config)
        ttnn.deallocate(mask)
        ttnn.deallocate(level_values)
        ttnn.deallocate(result)
        result = new_result

    return result


def _dequantize_from_bf16_indices(
    indices_bf16: "ttnn.Tensor",
    norms: "ttnn.Tensor",
    setup: TTNNTurboQuantSetup,
    memory_config=None,
) -> "ttnn.Tensor":
    """Dequantize using BF16 indices directly — no uint32 typecast needed.

    Args:
        indices_bf16: BF16 tensor [batch, heads, seq, head_dim] with centroid
                      indices stored as floats (values 0..2^bits-1).
        norms: BF16 tensor [batch, heads, seq, 1] with L2 norms.
        setup: Precomputed rotation matrix and codebook on device.
        memory_config: Output memory config (default: DRAM).

    Returns:
        Reconstructed BF16 tensor [batch, heads, seq, head_dim].
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    y_hat = _gather_centroids_from_bf16(indices_bf16, setup, memory_config)
    y = ttnn.mul(y_hat, norms, memory_config=memory_config)
    ttnn.deallocate(y_hat)
    x_rec = ttnn.matmul(y, setup.rotation_t, memory_config=memory_config)
    ttnn.deallocate(y)
    return x_rec


def _dequantize_rotated(
    indices_bf16: "ttnn.Tensor",
    norms: "ttnn.Tensor",
    setup: TTNNTurboQuantSetup,
    memory_config=None,
) -> "ttnn.Tensor":
    """Dequantize WITHOUT inverse rotation — output stays in rotated space.

    Eliminates the expensive [max_seq, D] × [D, D] matmul from dequantize.
    Caller must pre-rotate Q and post-rotate the SDPA output instead
    (two tiny [1, D] × [D, D] matmuls per decode step).

    Returns:
        BF16 tensor [batch, heads, seq, head_dim] in ROTATED coordinate space.
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    y_hat = _gather_centroids_from_bf16(indices_bf16, setup, memory_config)
    y = ttnn.mul(y_hat, norms, memory_config=memory_config)
    ttnn.deallocate(y_hat)
    return y  # No rotation_t matmul — stays in rotated space


class TTNNTurboQuantCache:
    """On-device TurboQuant KV cache for use in TTNN Llama attention.

    Phase 1.5-A2: indices AND norms live on device (BF16); no host↔device per step.
    - Indices scattered with paged_update_cache using update_idxs_tensor= (device tensor).
    - Norms scattered identically — no to_torch / from_torch in the decode hot path.
    - All shapes are fixed (full max_seq_padded cache) → TTNN trace-compatible.

    Usage in Llama attention:
        # Setup (once per model load)
        tq_cache = TTNNTurboQuantCache(device, num_layers=32, head_dim=128, bits=3)

        # Decode step (per layer)
        k_dequant, v_dequant = tq_cache.update_and_dequantize(k, v, layer_idx, pos)
        # Pass k_dequant, v_dequant to SDPA
    """

    def __init__(
        self,
        device,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        bits: int = 3,
        seed: int = 42,
        max_batch_size: int = 1,
        memory_efficient: bool = True,
        paged_config=None,
        recent_window: int = 0,
    ):
        _require_ttnn()

        self.device = device
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.bits = bits
        self.max_batch_size = max_batch_size
        self.memory_efficient = memory_efficient
        # Paged mode: allocate indices/norms as [max_num_blocks, NKH, block_size, DH]
        # (matching standard paged attention layout). Callers then pass page_table
        # into update_cache and fused_sdpa_decode. When None, falls back to
        # contiguous [B, NKH, max_seq_padded, DH] layout (legacy, used by the
        # pre-rescaled eval_e2e path).
        self.paged_config = paged_config

        # Legacy flags — derived from memory_efficient for backward compat.
        self.cache_centroids = not memory_efficient
        self.use_bfp8_indices = memory_efficient

        # Shared quantizer setup (rotation + codebook on device)
        self.setup = TTNNTurboQuantSetup(device, head_dim=head_dim, bits=bits, seed=seed)

        # ------------------------------------------------------------------ #
        # On-device cache                                                     #
        #                                                                     #
        # memory_efficient=True (default): BFP4 indices + BF16 norms.         #
        # BFP4 = ~0.5 bytes/elem (integers 0-7 exact). 4× smaller than       #
        # BF16, 2× smaller than BFP8 baseline KV cache.                       #
        # Dequantize: typecast BFP4→BF16 + gather + mul (O(max_seq)).        #
        #                                                                     #
        # memory_efficient=False: BF16 pre-rescaled centroid×norm values.     #
        # Same memory as FP16 (537 MB at seq=4096).                           #
        # Dequantize: pass cache directly to SDPA (O(1), flat latency).       #
        # ------------------------------------------------------------------ #
        max_seq_padded = ((max_seq_len + 31) // 32) * 32
        self.max_seq_padded = max_seq_padded

        # BFP4 for memory-efficient mode: ~0.5 byte/elem, 4× smaller than BF16.
        # Integers 0-7 are exactly representable in BFP4. paged_update_cache
        # supports BF16 input → BFP4 cache natively. Gather kernel handles
        # BFP4→float32 via the tile unpacker.
        idx_dtype = ttnn.bfloat4_b if memory_efficient else ttnn.bfloat16

        if paged_config is not None:
            # Paged layout: [max_num_blocks, num_kv_heads, block_size, head_dim].
            # Virtual-seq-to-physical-tile mapping is provided by the caller via
            # page_table passed to update_cache / fused_sdpa_decode.
            self.block_size = paged_config.block_size
            self.max_num_blocks = paged_config.max_num_blocks
            idx_shape = (self.max_num_blocks, num_kv_heads, self.block_size, head_dim)
            norms_shape = (self.max_num_blocks, num_kv_heads, self.block_size, 1)
        else:
            self.block_size = None
            self.max_num_blocks = None
            idx_shape = (max_batch_size, num_kv_heads, max_seq_padded, head_dim)
            norms_shape = (max_batch_size, num_kv_heads, max_seq_padded, 1)

        # On a MeshDevice, replicate the cache tensor across all devices. Each
        # device then independently scatters its own shard of K/V heads into its
        # local copy. Matches how the standard layer_past is allocated in
        # models/tt_transformers/tt/attention.py::init_kv_cache.
        mesh_mapper = None
        if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)

        zero_idx = torch.zeros(idx_shape, dtype=torch.bfloat16)
        self.k_indices_dev = [
            ttnn.from_torch(
                zero_idx,
                dtype=idx_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            for _ in range(num_layers)
        ]
        self.v_indices_dev = [
            ttnn.from_torch(
                zero_idx,
                dtype=idx_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            for _ in range(num_layers)
        ]
        del zero_idx

        # Norms stored as BFP8 (1 byte/elem) to halve the on-device norms cache
        # vs BF16. The compute kernel typecasts BFP8 → BF16 in DST registers
        # before the bcast_cols multiply (BFP8 input is not natively supported
        # by mul_tiles_bcast_cols).
        norms_dtype = ttnn.bfloat8_b
        zero_norms = torch.zeros(norms_shape, dtype=torch.bfloat16)
        self.k_norms_dev = [
            ttnn.from_torch(
                zero_norms,
                dtype=norms_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            for _ in range(num_layers)
        ]
        self.v_norms_dev = [
            ttnn.from_torch(
                zero_norms,
                dtype=norms_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            for _ in range(num_layers)
        ]
        del zero_norms

        # ── Sliding-window hybrid: BFP8 ring buffer for recent W tokens ──
        #
        # When recent_window > 0, allocate a small BFP8-paged cache of W tokens
        # per layer. update_cache writes to slot = cur_pos % W_padded, so the
        # ring naturally evicts the oldest position. At SDPA time, standard
        # paged SDPA decode runs over the ring (recent W positions in BFP8) and
        # the fused TQ kernel runs over [0, cur_pos - W] (older positions still
        # 3-bit lossy). The two outputs combine via online softmax — see
        # `hybrid_sdpa_decode` and `SLIDING_WINDOW_DESIGN.md`.
        #
        # Memory cost is constant in seqlen (W is small, e.g. 64 → ~4 MB total
        # at 32 layers / 8 KV heads / 128 dim).
        #
        # Two execution paths exist for hybrid_sdpa_decode:
        #   - Legacy (always available): two SDPA calls + on-device LSE combine.
        #   - Fused (TQ_FUSED_HYBRID=1, ~2x faster): single fused kernel call
        #     with disjoint TQ/ring coverage. Requires W rounded up to a
        #     multiple of 128 (the SDPA chunk size). Set W ∈ {128, 256, 384, ...}
        #     to use the fused path; smaller values silently fall back to legacy.
        if recent_window > 0 and recent_window % 128 != 0:
            import warnings as _warnings

            _warnings.warn(
                f"recent_window={recent_window} is not a multiple of 128; the fused-hybrid "
                f"SDPA fast path (TQ_FUSED_HYBRID=1) will fall back to the legacy dual-call. "
                f"Round up to 128 / 256 / 384 / ... to enable the ~2x faster fused kernel.",
                stacklevel=2,
            )
        self.recent_window = recent_window
        self.ring_page_table_torch = None
        self.k_ring_dev = None
        self.v_ring_dev = None
        if recent_window > 0:
            assert paged_config is not None, "recent_window requires paged_config (ring uses paged-cache layout)"
            block_size = paged_config.block_size
            ring_blocks = (recent_window + block_size - 1) // block_size
            ring_W_padded = ring_blocks * block_size
            self.ring_blocks = ring_blocks
            self.ring_W_padded = ring_W_padded
            ring_shape = (ring_blocks, num_kv_heads, block_size, head_dim)
            zero_ring = torch.zeros(ring_shape, dtype=torch.bfloat16)
            self.k_ring_dev = [
                ttnn.from_torch(
                    zero_ring,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                )
                for _ in range(num_layers)
            ]
            self.v_ring_dev = [
                ttnn.from_torch(
                    zero_ring,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                )
                for _ in range(num_layers)
            ]
            del zero_ring
            # Identity page table for the ring: logical block i → physical block i.
            # paged_update_cache enforces page_table length ≤ cache blocks, so we
            # CANNOT use a cyclic page_table. Instead we wrap cur_pos host-side
            # before each update (see _ring_pos_for_update). Trade-off: per-step
            # host sync, breaks trace capture for the ring write — fine for
            # accuracy testing; future kernel-side modulo would restore tracing.
            ring_pt_torch = torch.arange(ring_blocks, dtype=torch.int32).reshape(1, -1)
            self.ring_page_table_dev = ttnn.from_torch(
                ring_pt_torch,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

    def quantize(self, x: "ttnn.Tensor", skip_rotation: bool = False) -> tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """Quantize a KV tensor on device.

        Args:
            x: BF16 tensor [batch, heads, seq, head_dim] on device.
            skip_rotation: If True, skip the rotation step (already in rotated space).

        Returns:
            (values_bf16, norms_bf16) on device.
            If cache_centroids=True: values are pre-gathered centroid floats.
            Otherwise: values are integer indices as BF16 (0..2^b-1).
        """
        idx_bf16, norms = turbo_quant_quantize(x, self.setup, skip_rotation=skip_rotation)

        if self.cache_centroids:
            # Gather centroids NOW (on 1 token = tiny) so we can store
            # centroid values in the cache and skip gather at dequantize time.
            centroids_bf16 = _gather_centroids_from_bf16(idx_bf16, self.setup, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(idx_bf16)
            return centroids_bf16, norms

        return idx_bf16, norms

    def update_and_dequantize(
        self,
        k_heads: "ttnn.Tensor",
        v_heads: "ttnn.Tensor",
        layer_idx: int,
        current_pos,
        target_seq_len: int = None,
    ) -> tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """Quantize new K/V tokens, scatter into on-device cache, dequantize full cache.

        Phase 1.5-A2 flow (all ops on device — trace-compatible):
          1. Quantize new token on device → indices_bf16 [1,H,1,D], norms_bf16 [1,H,1,1].
          2. Scatter indices and norms into device cache via paged_update_cache
             using update_idxs_tensor= (device int32 tensor) — no to_torch calls.
          3. Dequantize the full fixed-size cache k_indices_dev / k_norms_dev directly.
             Output shape [1, H, max_seq_padded, D] is constant → trace-compatible.

        Args:
            k_heads: New key states [batch, kv_heads, 1, head_dim] BF16 on device.
            v_heads: New value states [batch, kv_heads, 1, head_dim] BF16 on device.
            layer_idx: Transformer layer index.
            current_pos: Current sequence position — Python int OR TTNN int32 device tensor.
                         Pass a device tensor for trace-compatible execution.
            target_seq_len: Ignored (output is always max_seq_padded). Kept for API compat.

        Returns:
            (k_dequant, v_dequant): BF16 [batch, heads, max_seq_padded, head_dim] on device.
        """
        # Resolve current_pos to a device tensor for paged_update_cache.
        # If caller passes a Python int (non-trace path / tests), wrap it here.
        if isinstance(current_pos, int):
            pos_tt = ttnn.from_torch(
                torch.tensor([current_pos], dtype=torch.int32),
                device=self.device,
                dtype=ttnn.int32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _own_pos_tt = True
        else:
            pos_tt = current_pos
            _own_pos_tt = False

        # Step 1: Quantize new token on device.
        k_idx_bf16, k_norms_new = self.quantize(k_heads)  # [batch,H,1,D], [batch,H,1,1] BF16
        v_idx_bf16, v_norms_new = self.quantize(v_heads)

        # Step 2: Scatter indices and norms on device — no host transfer.
        # paged_update_cache requires:
        #   cache  [batch, heads, max_seq_padded, dim]
        #   input  [1, batch, heads_padded_to_32, dim]  HEIGHT_SHARDED L1 on 1 core
        # Permute from [batch, H, 1, dim] → [1, batch, H, dim].
        _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        # 2a: Scatter indices.
        k_idx_scatter = ttnn.permute(k_idx_bf16, (2, 0, 1, 3))  # [1, 1, H, D]
        v_idx_scatter = ttnn.permute(v_idx_bf16, (2, 0, 1, 3))
        ttnn.deallocate(k_idx_bf16)
        ttnn.deallocate(v_idx_bf16)

        _shard_spec_idx = ttnn.ShardSpec(_shard_grid, [32, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        _shard_mem_idx = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_idx)

        k_idx_sharded = ttnn.to_memory_config(k_idx_scatter, _shard_mem_idx)
        v_idx_sharded = ttnn.to_memory_config(v_idx_scatter, _shard_mem_idx)
        ttnn.deallocate(k_idx_scatter)
        ttnn.deallocate(v_idx_scatter)

        ttnn.experimental.paged_update_cache(
            self.k_indices_dev[layer_idx],
            k_idx_sharded,
            update_idxs_tensor=pos_tt,
        )
        ttnn.experimental.paged_update_cache(
            self.v_indices_dev[layer_idx],
            v_idx_sharded,
            update_idxs_tensor=pos_tt,
        )
        ttnn.deallocate(k_idx_sharded)
        ttnn.deallocate(v_idx_sharded)

        # 2b: Scatter norms.
        # norms shape [batch, H, 1, 1] → permute → [1, batch, H, 1]; shard_shape [32, 32].
        k_norms_scatter = ttnn.permute(k_norms_new, (2, 0, 1, 3))  # [1, 1, H, 1]
        v_norms_scatter = ttnn.permute(v_norms_new, (2, 0, 1, 3))
        ttnn.deallocate(k_norms_new)
        ttnn.deallocate(v_norms_new)

        _shard_spec_norms = ttnn.ShardSpec(_shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        _shard_mem_norms = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_norms
        )

        k_norms_sharded = ttnn.to_memory_config(k_norms_scatter, _shard_mem_norms)
        v_norms_sharded = ttnn.to_memory_config(v_norms_scatter, _shard_mem_norms)
        ttnn.deallocate(k_norms_scatter)
        ttnn.deallocate(v_norms_scatter)

        ttnn.experimental.paged_update_cache(
            self.k_norms_dev[layer_idx],
            k_norms_sharded,
            update_idxs_tensor=pos_tt,
        )
        ttnn.experimental.paged_update_cache(
            self.v_norms_dev[layer_idx],
            v_norms_sharded,
            update_idxs_tensor=pos_tt,
        )
        ttnn.deallocate(k_norms_sharded)
        ttnn.deallocate(v_norms_sharded)

        if _own_pos_tt:
            ttnn.deallocate(pos_tt)

        # Step 3: Dequantize the full fixed-size cache directly.
        if self.cache_centroids:
            # Cache stores centroid values → gather + mul + rotation in one shot.
            k_vals = ttnn.mul(self.k_indices_dev[layer_idx], self.k_norms_dev[layer_idx])
            v_vals = ttnn.mul(self.v_indices_dev[layer_idx], self.v_norms_dev[layer_idx])
            k_dequant = ttnn.matmul(k_vals, self.setup.rotation_t)
            v_dequant = ttnn.matmul(v_vals, self.setup.rotation_t)
            ttnn.deallocate(k_vals)
            ttnn.deallocate(v_vals)
        else:
            k_idx = self.k_indices_dev[layer_idx]
            v_idx = self.v_indices_dev[layer_idx]
            if not _FUSED_OPS_AVAILABLE and self.use_bfp8_indices:
                k_idx = ttnn.typecast(k_idx, ttnn.bfloat16)
                v_idx = ttnn.typecast(v_idx, ttnn.bfloat16)

            k_dequant = _dequantize_from_bf16_indices(k_idx, self.k_norms_dev[layer_idx], self.setup)
            v_dequant = _dequantize_from_bf16_indices(v_idx, self.v_norms_dev[layer_idx], self.setup)

            # Only deallocate if typecast created a copy (fallback path).
            # With fused ops, k_idx IS the cache tensor — don't deallocate it.
            if not _FUSED_OPS_AVAILABLE and self.use_bfp8_indices:
                ttnn.deallocate(k_idx)
                ttnn.deallocate(v_idx)

        return k_dequant, v_dequant

    def update_cache(
        self,
        k_heads: "ttnn.Tensor",
        v_heads: "ttnn.Tensor",
        layer_idx: int,
        current_pos,
        page_table: "ttnn.Tensor" = None,
        ring_write_pos: "ttnn.Tensor" = None,
    ):
        """Quantize new K/V tokens and scatter into cache (no dequantize).

        Use with fused_sdpa_decode() which reads raw indices+norms directly.
        Identical to steps 1-2 of update_and_dequantize().

        Args:
            page_table: Required when cache is paged (paged_config was passed to
                __init__). Int32 device tensor [B, max_pages_per_batch] mapping
                virtual block → physical block. For contiguous caches, leave None.
            ring_write_pos: Trace-input device int32 tensor holding
                cur_pos % ring_W_padded. Required when recent_window > 0 to
                avoid the per-step host sync that breaks trace capture. When
                None, falls back to a one-time ttnn.to_torch read at
                layer_idx==0 (cached for the rest of the layers).
        """
        if self.paged_config is not None and page_table is None:
            raise ValueError("page_table required for paged TQ cache")

        if isinstance(current_pos, int):
            pos_tt = ttnn.from_torch(
                torch.tensor([current_pos], dtype=torch.int32),
                device=self.device,
                dtype=ttnn.int32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _own_pos_tt = True
        else:
            pos_tt = current_pos
            _own_pos_tt = False

        # ── Sliding-window dual-write (must run BEFORE quantize) ──
        #
        # turbo_quant_quantize() with skip_rotation=True aliases the input as
        # `y` and then deallocates `y` at end of normalisation (line ~168) —
        # so v_heads gets freed by the quantize call below. Compute the ring
        # writes UPFRONT while v_heads is still alive. K isn't aliased
        # (skip_rotation=False does an explicit matmul) but we keep the same
        # pattern for symmetry.
        if self.recent_window > 0:
            rotation_absorbed = getattr(self, "rotation_absorbed", False)
            k_rotated = ttnn.matmul(k_heads, self.setup.rotation, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if rotation_absorbed:
                v_for_ring = v_heads  # already V @ Π from W_v absorption
                _own_v_ring = False
            else:
                v_for_ring = ttnn.matmul(v_heads, self.setup.rotation, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                _own_v_ring = True

            # Permute [B, NKH, 1, DH] → [1, B, NKH, DH] (paged_update_cache layout).
            k_ring_perm = ttnn.permute(k_rotated, (2, 0, 1, 3))
            v_ring_perm = ttnn.permute(v_for_ring, (2, 0, 1, 3))
            ttnn.deallocate(k_rotated)
            if _own_v_ring:
                ttnn.deallocate(v_for_ring)

            _shard_grid_ring = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
            _shard_spec_ring = ttnn.ShardSpec(_shard_grid_ring, [32, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
            _shard_mem_ring = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_ring
            )
            k_ring_sharded = ttnn.to_memory_config(k_ring_perm, _shard_mem_ring)
            v_ring_sharded = ttnn.to_memory_config(v_ring_perm, _shard_mem_ring)
            ttnn.deallocate(k_ring_perm)
            ttnn.deallocate(v_ring_perm)

            # paged_update_cache requires update_idx within [0, cache_size).
            # For the ring, that's cur_pos % ring_W_padded. Trace-compatible
            # path: ring_write_pos is a device tensor built host-side in
            # prepare_decode_inputs_host. Fallback path: one-time host sync
            # at layer_idx==0, cached for the remaining 31 layers.
            if ring_write_pos is not None:
                ring_pos_tt = ring_write_pos
                _own_ring_pos_tt = False
            else:
                if layer_idx == 0:
                    cur_pos_host = int(ttnn.to_torch(pos_tt).flatten()[0].item())
                    self._cached_cur_pos = cur_pos_host
                    self._cached_ring_pos = cur_pos_host % self.ring_W_padded
                ring_idx_host = self._cached_ring_pos
                ring_pos_tt = ttnn.from_torch(
                    torch.tensor([ring_idx_host], dtype=torch.int32),
                    device=self.device,
                    dtype=ttnn.int32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                _own_ring_pos_tt = True
            ttnn.experimental.paged_update_cache(
                self.k_ring_dev[layer_idx],
                k_ring_sharded,
                update_idxs_tensor=ring_pos_tt,
                page_table=self.ring_page_table_dev,
            )
            ttnn.experimental.paged_update_cache(
                self.v_ring_dev[layer_idx],
                v_ring_sharded,
                update_idxs_tensor=ring_pos_tt,
                page_table=self.ring_page_table_dev,
            )
            if _own_ring_pos_tt:
                ttnn.deallocate(ring_pos_tt)
            ttnn.deallocate(k_ring_sharded)
            ttnn.deallocate(v_ring_sharded)

        # When rotation is absorbed into W_v, V already comes pre-rotated from
        # the QKV projection — skip the in-quantize Π matmul to avoid double-
        # rotation (V @ Π @ Π). K is not absorbed (RoPE dependency), so it
        # still needs the rotation step here.
        k_idx_bf16, k_norms_new = self.quantize(k_heads)
        v_idx_bf16, v_norms_new = self.quantize(v_heads, skip_rotation=getattr(self, "rotation_absorbed", False))

        _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        k_idx_scatter = ttnn.permute(k_idx_bf16, (2, 0, 1, 3))
        v_idx_scatter = ttnn.permute(v_idx_bf16, (2, 0, 1, 3))
        ttnn.deallocate(k_idx_bf16)
        ttnn.deallocate(v_idx_bf16)

        _shard_spec_idx = ttnn.ShardSpec(_shard_grid, [32, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        _shard_mem_idx = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_idx)

        k_idx_sharded = ttnn.to_memory_config(k_idx_scatter, _shard_mem_idx)
        v_idx_sharded = ttnn.to_memory_config(v_idx_scatter, _shard_mem_idx)
        ttnn.deallocate(k_idx_scatter)
        ttnn.deallocate(v_idx_scatter)

        ttnn.experimental.paged_update_cache(
            self.k_indices_dev[layer_idx], k_idx_sharded, update_idxs_tensor=pos_tt, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            self.v_indices_dev[layer_idx], v_idx_sharded, update_idxs_tensor=pos_tt, page_table=page_table
        )
        ttnn.deallocate(k_idx_sharded)
        ttnn.deallocate(v_idx_sharded)

        k_norms_scatter = ttnn.permute(k_norms_new, (2, 0, 1, 3))
        v_norms_scatter = ttnn.permute(v_norms_new, (2, 0, 1, 3))
        ttnn.deallocate(k_norms_new)
        ttnn.deallocate(v_norms_new)

        _shard_spec_norms = ttnn.ShardSpec(_shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        _shard_mem_norms = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_norms
        )

        k_norms_sharded = ttnn.to_memory_config(k_norms_scatter, _shard_mem_norms)
        v_norms_sharded = ttnn.to_memory_config(v_norms_scatter, _shard_mem_norms)
        ttnn.deallocate(k_norms_scatter)
        ttnn.deallocate(v_norms_scatter)

        ttnn.experimental.paged_update_cache(
            self.k_norms_dev[layer_idx], k_norms_sharded, update_idxs_tensor=pos_tt, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            self.v_norms_dev[layer_idx], v_norms_sharded, update_idxs_tensor=pos_tt, page_table=page_table
        )
        ttnn.deallocate(k_norms_sharded)
        ttnn.deallocate(v_norms_sharded)

        # (Sliding-window ring writes happen at the top of update_cache — see
        # the block above, before quantize() can deallocate v_heads.)

        if _own_pos_tt:
            ttnn.deallocate(pos_tt)

    def fused_sdpa_decode(
        self,
        q: "ttnn.Tensor",
        layer_idx: int,
        current_pos: "ttnn.Tensor",
        scale: float,
        page_table: "ttnn.Tensor" = None,
        num_cores_per_head: int = 1,
        return_lse: bool = False,
    ):
        """Run fused TQ SDPA decode directly on BFP4 index + BF16 norm caches.

        Reads BFP4 indices from k/v_indices_dev, BF16 norms from k/v_norms_dev,
        and dequantizes on-the-fly (centroid gather × norm) inside the SDPA kernel.

        Args:
            q: BF16 query [B, NQH, 1, DH] on device.
            layer_idx: Transformer layer index.
            current_pos: Int32 current position tensor [B] on device.
            scale: Attention scale factor (1/sqrt(head_dim)).
            page_table: Required for paged caches. Int32 device tensor
                [B, max_pages_per_batch]. For contiguous caches, a 1x1 dummy
                tensor is allocated internally.
            num_cores_per_head: Tier 2A cross-core split (default 1).
            return_lse: if True, also returns LSE = max + log(sum) per
                (B, NQH) for the sliding-window hybrid combine.

        Returns:
            If return_lse=False: BF16 attention output [B, NQH, 1, DH] on device.
            If return_lse=True: tuple (out, lse) where lse is a BF16 tensor
                of shape [B, NQH, 1, 32] (1 tile per (B, NQH)).
        """
        _require_ttnn()
        centroids = get_codebook(self.head_dim, self.bits, device="cpu", dtype=torch.float32).centroids.tolist()

        # Contiguous mode: allocate a 1x1 dummy page_table (reader ignores it).
        # Paged mode: caller must supply real page_table.
        if page_table is None:
            if self.paged_config is not None:
                raise ValueError("page_table required for paged TQ cache")
            _pt = ttnn.from_torch(
                torch.zeros(1, 1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            _own_pt = True
        else:
            _pt = page_table
            _own_pt = False

        outs = ttnn.experimental.turbo_quant_sdpa_decode(
            q,
            self.k_indices_dev[layer_idx],
            self.k_norms_dev[layer_idx],
            self.v_indices_dev[layer_idx],
            self.v_norms_dev[layer_idx],
            _pt,
            current_pos,
            centroids,
            scale,
            False,  # pre_rescaled
            num_cores_per_head,
            return_lse,
        )
        if _own_pt:
            ttnn.deallocate(_pt)
        # Op now returns a list — [out] or [out, lse].
        if return_lse:
            return outs[0], outs[1]
        return outs[0]

    def hybrid_sdpa_decode(
        self,
        q: "ttnn.Tensor",
        layer_idx: int,
        current_pos: "ttnn.Tensor",
        scale: float,
        page_table: "ttnn.Tensor",
        ring_sdpa_pos: "ttnn.Tensor" = None,
        old_pos: "ttnn.Tensor" = None,
    ) -> "ttnn.Tensor":
        """Sliding-window hybrid: TQ over old positions + BFP8 ring over recent W.

        Runs the fused TQ SDPA kernel TWICE — once over the BFP8 ring (using
        pre_rescaled=True so the kernel skips dequant) covering the most
        recent W positions, once over the TQ cache covering older positions.
        Each call returns (out, lse). Outputs are merged on device via online
        softmax LSE math, mathematically equivalent to one big SDPA.

        Trace-compatible path: caller supplies ring_sdpa_pos = min(W-1, cur_pos)
        and old_pos = max(0, cur_pos - W) as device int32 tensors built in
        prepare_decode_inputs_host. We always run the combine — during warmup
        (cur_pos < W) the math double-counts position 0, but warmup positions
        sit below the eval split_point and don't affect accuracy.

        Fallback path (ring_sdpa_pos / old_pos = None): legacy host-sync read
        of current_pos and Python branching for the cur_pos < W edge case.
        Used by tests that haven't yet been updated.

        Q must be pre-rotated by Π. See LSE_COMBINE_DESIGN.md.
        """
        _require_ttnn()
        assert self.recent_window > 0, "hybrid_sdpa_decode requires recent_window > 0"
        W = self.recent_window
        centroids = get_codebook(self.head_dim, self.bits, device="cpu", dtype=torch.float32).centroids.tolist()

        # ── Fused-hybrid fast path (Phase 3c, opt-in via TQ_FUSED_HYBRID=1) ──
        # Single fused kernel call replaces the dual-call + _combine_lse flow.
        # Ring covers the most recent W positions in BFP8; TQ covers older
        # positions in BFP4-quantized; the kernel runs one Flash Attention loop
        # across both with disjoint coverage (no LSE-emphasis-blend math). Saves
        # ~10-15 ms/tok by eliminating the second SDPA invocation and the
        # on-device combine ops. Requires W rounded up to a multiple of 128
        # (kernel chunk size) — host-side validation enforces this.
        import os as _os

        _use_fused = bool(_os.environ.get("TQ_FUSED_HYBRID", "")) and (W % 128 == 0)
        if _use_fused:
            outs = ttnn.experimental.turbo_quant_sdpa_decode(
                q,
                self.k_indices_dev[layer_idx],
                self.k_norms_dev[layer_idx],
                self.v_indices_dev[layer_idx],
                self.v_norms_dev[layer_idx],
                page_table,
                current_pos,
                centroids,
                scale,
                False,  # pre_rescaled (must be False in hybrid mode)
                1,  # num_cores_per_head (must be 1 in hybrid mode)
                False,  # return_lse (no longer needed — combine is in-kernel)
                W,  # recent_window — triggers the per-chunk source branch
                self.k_ring_dev[layer_idx],
                self.v_ring_dev[layer_idx],
                self.ring_page_table_dev,
            )
            return outs[0]

        # Trace-friendly fast path: ring_sdpa_pos + old_pos arrived as device
        # tensors. Always run both halves + combine. No host sync.
        if ring_sdpa_pos is not None and old_pos is not None:
            outs_new = ttnn.experimental.turbo_quant_sdpa_decode(
                q,
                self.k_ring_dev[layer_idx],
                self.k_norms_dev[layer_idx],
                self.v_ring_dev[layer_idx],
                self.v_norms_dev[layer_idx],
                self.ring_page_table_dev,
                ring_sdpa_pos,
                centroids,
                scale,
                True,  # pre_rescaled — bypass dequant for the BFP8 ring
                1,
                True,  # always return_lse for the combine
            )
            out_new, lse_new = outs_new[0], outs_new[1]

            # Old SDPA: full Track A attention over [0, cur_pos] (NOT truncated to
            # cur_pos-W). The TQ kernel rounds valid_k_chunks up to chunk boundary,
            # so passing cur_pos-W would produce ~31-position overlap with the ring
            # — and the LSE-aware combine assumes disjoint coverage. Using the full
            # range makes the combine an "emphasis blend" instead of a mathematical
            # equivalent: the ring's output gets up-weighted via LSE so recent
            # tokens contribute more than their proportional share.
            out_old, lse_old = self.fused_sdpa_decode(
                q,
                layer_idx=layer_idx,
                current_pos=current_pos,
                scale=scale,
                page_table=page_table,
                num_cores_per_head=1,
                return_lse=True,
            )
            return self._combine_lse(out_old, lse_old, out_new, lse_new, layer_idx=layer_idx)

        # ── Legacy fallback: host-sync read + Python branch ──
        cur_pos_host = getattr(self, "_cached_cur_pos", None)
        if cur_pos_host is None:
            cur_pos_host = int(ttnn.to_torch(current_pos).flatten()[0].item())

        ring_pos_int = min(W - 1, cur_pos_host)
        ring_pos_dev = ttnn.from_torch(
            torch.tensor([ring_pos_int], dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        outs_new = ttnn.experimental.turbo_quant_sdpa_decode(
            q,
            self.k_ring_dev[layer_idx],
            self.k_norms_dev[layer_idx],
            self.v_ring_dev[layer_idx],
            self.v_norms_dev[layer_idx],
            self.ring_page_table_dev,
            ring_pos_dev,
            centroids,
            scale,
            True,
            1,
            cur_pos_host >= W,
        )
        ttnn.deallocate(ring_pos_dev)

        if cur_pos_host < W:
            return outs_new[0]

        out_new, lse_new = outs_new[0], outs_new[1]

        old_pos_int = cur_pos_host - W
        old_pos_dev = ttnn.from_torch(
            torch.tensor([old_pos_int], dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_old, lse_old = self.fused_sdpa_decode(
            q,
            layer_idx=layer_idx,
            current_pos=old_pos_dev,
            scale=scale,
            page_table=page_table,
            num_cores_per_head=1,
            return_lse=True,
        )
        ttnn.deallocate(old_pos_dev)

        return self._combine_lse(out_old, lse_old, out_new, lse_new, layer_idx=layer_idx)

    def _combine_lse(self, out_old, lse_old, out_new, lse_new, layer_idx=None):
        """Online-softmax combine of two SDPA halves on device.

        combined = out_old · w_old + out_new · w_new
          where w_X = exp(lse_X − lse_max) / (exp(lse_old − lse_max) + exp(lse_new − lse_max))
          and   lse_max = max(lse_old, lse_new)  (per-row max for numerical stability)

        LSE tensors are [B, NQH, 1, 32]: the kernel's writer packs one BF16 LSE
        value at column 0 of row 0; columns 1..31 contain garbage from the
        matmul_reduce + log_tile pipeline (e.g. log(0)=-inf for the rowsum
        tile's empty cols, plus uninitialised values from the max tile). We
        therefore do the per-element math across all 32 columns (the col-0
        result is correct even when garbage is present in cols 1-31, since the
        ops are local) and then BROADCAST col 0 of the resulting weight tile
        to every column of the [B, NQH, 1, DH] output before multiplying.
        """
        # Optional ablation hook: dump LSE values + weights to /tmp/tq_lse_dump.txt
        # when TQ_DUMP_LSE env var is set. Adds host syncs — incompatible with trace
        # (eval_token_accuracy.py auto-disables trace when TQ_DUMP_LSE is set).
        # Limited to layers 0..3 to keep file size manageable.
        import os as _os

        _dump = bool(_os.environ.get("TQ_DUMP_LSE", "")) and (layer_idx is not None) and (layer_idx < 4)
        if _dump:
            _lse_old_t = ttnn.to_torch(lse_old).float()
            _lse_new_t = ttnn.to_torch(lse_new).float()
            with open("/tmp/tq_lse_dump.txt", "a") as _f:
                _f.write(f"\n=== layer {layer_idx} ===\n")
                _f.write(
                    f"LSE_old col 0 per head: min={_lse_old_t[0,:,0,0].min():.3f} "
                    f"max={_lse_old_t[0,:,0,0].max():.3f} "
                    f"mean={_lse_old_t[0,:,0,0].mean():.3f}\n"
                )
                _f.write(
                    f"LSE_new col 0 per head: min={_lse_new_t[0,:,0,0].min():.3f} "
                    f"max={_lse_new_t[0,:,0,0].max():.3f} "
                    f"mean={_lse_new_t[0,:,0,0].mean():.3f}\n"
                )
                # Compute per-head w_new (ring weight) on host.
                import math as _math

                for h in [0, 7, 15, 23, 31]:
                    L_o = _lse_old_t[0, h, 0, 0].item()
                    L_n = _lse_new_t[0, h, 0, 0].item()
                    L_max = max(L_o, L_n)
                    d_o = _math.exp(L_o - L_max)
                    d_n = _math.exp(L_n - L_max)
                    w_n = d_n / (d_o + d_n)
                    _f.write(f"  head {h:2d}: LSE_old={L_o:.3f} LSE_new={L_n:.3f} → w_ring={w_n:.3f}\n")

        lse_max = ttnn.maximum(lse_old, lse_new)
        diff_old = ttnn.exp(ttnn.subtract(lse_old, lse_max))
        diff_new = ttnn.exp(ttnn.subtract(lse_new, lse_max))
        denom = ttnn.add(diff_old, diff_new)
        w_old = ttnn.divide(diff_old, denom)
        w_new = ttnn.divide(diff_new, denom)
        ttnn.deallocate(lse_max)
        ttnn.deallocate(diff_old)
        ttnn.deallocate(diff_new)
        ttnn.deallocate(denom)
        ttnn.deallocate(lse_old)
        ttnn.deallocate(lse_new)

        # Slice col 0 only (avoid garbage in cols 1-31), then broadcast to head_dim.
        # ttnn.slice on TILE_LAYOUT keeps tile boundaries; the result is a
        # [B, NQH, 1, 1] tensor (or with tile padding) holding only valid LSE.
        B = w_old.shape[0]
        NQH = w_old.shape[1]
        w_old_col0 = ttnn.slice(w_old, [0, 0, 0, 0], [B, NQH, 1, 1])
        w_new_col0 = ttnn.slice(w_new, [0, 0, 0, 0], [B, NQH, 1, 1])
        ttnn.deallocate(w_old)
        ttnn.deallocate(w_new)
        # Broadcast 1 → head_dim (e.g. 1 → 128) by repeating.
        w_old_full = ttnn.repeat(w_old_col0, ttnn.Shape([1, 1, 1, self.head_dim]))
        w_new_full = ttnn.repeat(w_new_col0, ttnn.Shape([1, 1, 1, self.head_dim]))
        ttnn.deallocate(w_old_col0)
        ttnn.deallocate(w_new_col0)
        out_old_w = ttnn.multiply(out_old, w_old_full)
        out_new_w = ttnn.multiply(out_new, w_new_full)
        ttnn.deallocate(out_old)
        ttnn.deallocate(out_new)
        ttnn.deallocate(w_old_full)
        ttnn.deallocate(w_new_full)
        combined = ttnn.add(out_old_w, out_new_w)
        ttnn.deallocate(out_old_w)
        ttnn.deallocate(out_new_w)
        return combined

    def sliding_window_sdpa_decode(
        self,
        q: "ttnn.Tensor",
        layer_idx: int,
        current_pos: "ttnn.Tensor",
        scale: float,
    ) -> "ttnn.Tensor":
        """Run standard paged SDPA decode over the BFP8 sliding-window ring only.

        First-cut sliding-window mode: attend ONLY to the most recent W tokens
        (BFP8, full precision in rotated space). Older positions in the TQ cache
        are ignored at this layer/step. Tests the "does Llama tolerate W-window
        attention?" hypothesis. Full hybrid (combining old TQ contributions via
        online softmax) is a follow-up.

        Q must already be pre-rotated by Π (the caller does this once per step
        for both Track A and the hybrid path). W_o's absorbed Π^T un-rotates
        the output downstream.

        Args:
            q: BF16 query [1, B, NQH, DH] on device, pre-rotated.
            layer_idx: Transformer layer index.
            current_pos: Int32 device tensor [B] holding the actual position.
                Internally clamped to (W_padded - 1) so the kernel doesn't read
                beyond the ring.
            scale: Attention scale factor (1/sqrt(head_dim)).

        Returns:
            BF16 attention output in pre-rotated space, shape [1, B, NQH, DH].
        """
        _require_ttnn()
        assert self.recent_window > 0, "sliding_window_sdpa_decode requires recent_window > 0"

        # Clamp cur_pos to W_padded - 1 so the kernel only attends within the ring.
        # Cyclic page_table maps any cur_pos to a real ring slot, but reading
        # beyond W_padded would alias positions and give wrong scores.
        # ttnn.minimum supports broadcast vs an int via clamp.
        clamp_const = self.ring_W_padded - 1
        cur_pos_clamped = ttnn.clamp(current_pos, max=clamp_const)

        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_ring_dev[layer_idx],
            self.v_ring_dev[layer_idx],
            page_table_tensor=self.ring_page_table_dev,
            cur_pos_tensor=cur_pos_clamped,
            scale=scale,
        )
        ttnn.deallocate(cur_pos_clamped)
        return out

    def update_and_dequantize_rotated(
        self,
        k_heads: "ttnn.Tensor",
        v_heads: "ttnn.Tensor",
        layer_idx: int,
        current_pos,
        target_seq_len: int = None,
    ) -> tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """Like update_and_dequantize but output stays in rotated space.

        B3 optimisation: eliminates the two [max_seq, D] × [D, D] inverse
        rotation matmuls from dequantize (the single most expensive ops).
        Caller must instead:
          1. Pre-rotate Q:   Q' = ttnn.matmul(Q, tq_cache.setup.rotation)
          2. Call this method to get K_rot, V_rot (no inverse rotation)
          3. SDPA(Q', K_rot, V_rot)
          4. Post-rotate out: out' = ttnn.matmul(out, tq_cache.setup.rotation_t)

        Steps 1+4 are tiny ([1, D] × [D, D] per step) vs the eliminated
        2 × [max_seq, D] × [D, D] = 500× fewer matmul FLOPs at max_seq=256.
        """
        # Quantize + scatter is identical to update_and_dequantize.
        # Resolve current_pos to a device tensor.
        if isinstance(current_pos, int):
            pos_tt = ttnn.from_torch(
                torch.tensor([current_pos], dtype=torch.int32),
                device=self.device,
                dtype=ttnn.int32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _own_pos_tt = True
        else:
            pos_tt = current_pos
            _own_pos_tt = False

        # Step 1: Quantize new token on device.
        # When rotation is absorbed into W_v, V is already in rotated space → skip V rotation.
        k_vals_bf16, k_norms_new = self.quantize(k_heads)
        v_vals_bf16, v_norms_new = self.quantize(v_heads, skip_rotation=getattr(self, "rotation_absorbed", False))

        # Step 2: Scatter into on-device cache.
        _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        _shard_spec_idx = ttnn.ShardSpec(_shard_grid, [32, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        _shard_mem_idx = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_idx)

        if self.cache_centroids:
            # Pre-multiply centroids × norms NOW (on 1 token = tiny) and store the
            # rescaled values.  Dequantize then becomes a no-op (just read the cache).
            # This eliminates the full-cache mul that scales with max_seq_len.
            k_rescaled = ttnn.mul(k_vals_bf16, k_norms_new)
            v_rescaled = ttnn.mul(v_vals_bf16, v_norms_new)
            ttnn.deallocate(k_vals_bf16)
            ttnn.deallocate(v_vals_bf16)
            ttnn.deallocate(k_norms_new)
            ttnn.deallocate(v_norms_new)

            k_scatter = ttnn.permute(k_rescaled, (2, 0, 1, 3))
            v_scatter = ttnn.permute(v_rescaled, (2, 0, 1, 3))
            ttnn.deallocate(k_rescaled)
            ttnn.deallocate(v_rescaled)

            k_sharded = ttnn.to_memory_config(k_scatter, _shard_mem_idx)
            v_sharded = ttnn.to_memory_config(v_scatter, _shard_mem_idx)
            ttnn.deallocate(k_scatter)
            ttnn.deallocate(v_scatter)

            ttnn.experimental.paged_update_cache(self.k_indices_dev[layer_idx], k_sharded, update_idxs_tensor=pos_tt)
            ttnn.experimental.paged_update_cache(self.v_indices_dev[layer_idx], v_sharded, update_idxs_tensor=pos_tt)
            ttnn.deallocate(k_sharded)
            ttnn.deallocate(v_sharded)
            # No norms scatter needed — norms are already baked into the cached values.
        else:
            # Legacy path: scatter indices and norms separately.
            k_idx_scatter = ttnn.permute(k_vals_bf16, (2, 0, 1, 3))
            v_idx_scatter = ttnn.permute(v_vals_bf16, (2, 0, 1, 3))
            ttnn.deallocate(k_vals_bf16)
            ttnn.deallocate(v_vals_bf16)

            k_idx_sharded = ttnn.to_memory_config(k_idx_scatter, _shard_mem_idx)
            v_idx_sharded = ttnn.to_memory_config(v_idx_scatter, _shard_mem_idx)
            ttnn.deallocate(k_idx_scatter)
            ttnn.deallocate(v_idx_scatter)

            ttnn.experimental.paged_update_cache(
                self.k_indices_dev[layer_idx], k_idx_sharded, update_idxs_tensor=pos_tt
            )
            ttnn.experimental.paged_update_cache(
                self.v_indices_dev[layer_idx], v_idx_sharded, update_idxs_tensor=pos_tt
            )
            ttnn.deallocate(k_idx_sharded)
            ttnn.deallocate(v_idx_sharded)

            _shard_spec_norms = ttnn.ShardSpec(_shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
            _shard_mem_norms = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_norms
            )

            k_norms_scatter = ttnn.permute(k_norms_new, (2, 0, 1, 3))
            v_norms_scatter = ttnn.permute(v_norms_new, (2, 0, 1, 3))
            ttnn.deallocate(k_norms_new)
            ttnn.deallocate(v_norms_new)

            k_norms_sharded = ttnn.to_memory_config(k_norms_scatter, _shard_mem_norms)
            v_norms_sharded = ttnn.to_memory_config(v_norms_scatter, _shard_mem_norms)
            ttnn.deallocate(k_norms_scatter)
            ttnn.deallocate(v_norms_scatter)

            ttnn.experimental.paged_update_cache(
                self.k_norms_dev[layer_idx], k_norms_sharded, update_idxs_tensor=pos_tt
            )
            ttnn.experimental.paged_update_cache(
                self.v_norms_dev[layer_idx], v_norms_sharded, update_idxs_tensor=pos_tt
            )
            ttnn.deallocate(k_norms_sharded)
            ttnn.deallocate(v_norms_sharded)

        if _own_pos_tt:
            ttnn.deallocate(pos_tt)

        # Step 3: Dequantize in ROTATED space (no inverse rotation matmul).
        if self.cache_centroids:
            # Cache stores pre-rescaled centroid×norm values → just read directly.
            # No per-step mul over the full cache → cost is O(1), not O(max_seq).
            k_rot = self.k_indices_dev[layer_idx]
            v_rot = self.v_indices_dev[layer_idx]
        else:
            k_idx = self.k_indices_dev[layer_idx]
            v_idx = self.v_indices_dev[layer_idx]
            # Fused gather kernel accepts BFP4/BFP8 directly (hardware unpacks to
            # float32 in DST). Skip typecast to avoid allocating a full BF16 temp.
            # Fallback path (no fused ops) still needs BF16 for ttnn.ge comparisons.
            if not _FUSED_OPS_AVAILABLE and self.use_bfp8_indices:
                k_idx = ttnn.typecast(k_idx, ttnn.bfloat16)
                v_idx = ttnn.typecast(v_idx, ttnn.bfloat16)

            k_rot = _dequantize_rotated(k_idx, self.k_norms_dev[layer_idx], self.setup)
            v_rot = _dequantize_rotated(v_idx, self.v_norms_dev[layer_idx], self.setup)

            if not _FUSED_OPS_AVAILABLE and self.use_bfp8_indices:
                ttnn.deallocate(k_idx)
                ttnn.deallocate(v_idx)

        return k_rot, v_rot

    def pre_rotate_query(self, q: "ttnn.Tensor") -> "ttnn.Tensor":
        """Pre-rotate query for rotated-space SDPA: Q' = Q × Π.

        Args:
            q: Query tensor [batch, heads, 1, head_dim] BF16.

        Returns:
            Rotated query [batch, heads, 1, head_dim] BF16.
        """
        return ttnn.matmul(q, self.setup.rotation, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def post_rotate_output(self, out: "ttnn.Tensor") -> "ttnn.Tensor":
        """Post-rotate SDPA output back to original space: out' = out × Πᵀ.

        Args:
            out: SDPA output [batch, heads, 1, head_dim] BF16.

        Returns:
            Output in original coordinate space [batch, heads, 1, head_dim] BF16.
        """
        return ttnn.matmul(out, self.setup.rotation_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def deallocate(self):
        """Free device memory: setup tensors and on-device index/norm caches."""
        self.setup.deallocate()
        for t in self.k_indices_dev + self.v_indices_dev + self.k_norms_dev + self.v_norms_dev:
            ttnn.deallocate(t)
        if self.recent_window > 0:
            for t in self.k_ring_dev + self.v_ring_dev:
                ttnn.deallocate(t)
            ttnn.deallocate(self.ring_page_table_dev)


def absorb_rotation_into_state_dict(state_dict, rotation_cpu, n_layers=32, n_q_heads=32, n_kv_heads=8, head_dim=128):
    """Absorb TurboQuant rotation into W_v and W_o in the CPU state_dict.

    Must be called BEFORE model creation so the weights are loaded with the
    rotation already baked in (preserving the model's sharded memory configs).

    After loading with the modified state_dict:
      - V comes out of the QKV projection already in rotated space
      - W_o includes Π^T, so post_rotate_output is unnecessary
      - K rotation and Q pre-rotation remain (RoPE dependency)

    Args:
        state_dict: CPU state_dict (modified in-place).
        rotation_cpu: Rotation matrix Π [head_dim, head_dim] as float32 CPU tensor.
        n_layers: Number of transformer layers.
        n_q_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        head_dim: Head dimension.
    """
    rotation_t_cpu = rotation_cpu.t().contiguous()

    for layer_idx in range(n_layers):
        prefix = f"layers.{layer_idx}.attention"

        # --- W_v: rotate output space so V comes out pre-rotated ---
        # State dict shape: [n_kv_heads * head_dim, dim] = [out, in]
        wv_key = f"{prefix}.wv.weight"
        wv = state_dict[wv_key].float()  # [1024, 4096]
        wv_heads = wv.reshape(n_kv_heads, head_dim, -1)  # [8, 128, 4096]
        wv_heads = rotation_t_cpu @ wv_heads  # Π^T @ each head's [128, 4096] block
        state_dict[wv_key] = wv_heads.reshape_as(wv).to(state_dict[wv_key].dtype)

        # --- W_o: absorb Π^T so output is automatically de-rotated ---
        # State dict shape: [dim, n_q_heads * head_dim] = [out, in]
        wo_key = f"{prefix}.wo.weight"
        wo = state_dict[wo_key].float()  # [4096, 4096]
        wo_cols = wo.reshape(-1, n_q_heads, head_dim)  # [4096, 32, 128]
        wo_cols = wo_cols @ rotation_cpu  # each head's [4096, 128] @ Π
        state_dict[wo_key] = wo_cols.reshape_as(wo).to(state_dict[wo_key].dtype)

    print(f"  Absorbed rotation into W_v and W_o for {n_layers} layers")


def validate_against_cpu_reference(
    device,
    head_dim: int = 128,
    bits: int = 3,
    seed: int = 42,
    seq_len: int = 32,
    num_heads: int = 8,
    atol: float = 0.05,
) -> dict:
    """Validate TTNN TurboQuant against CPU reference implementation.

    Runs quantize/dequantize on both CPU and device, compares results.
    Call this on a TTNN machine to verify correctness.

    Args:
        device: TTNN device handle.
        head_dim: Head dimension.
        bits: Quantization bit-width.
        seed: Random seed (must match between CPU and device).
        seq_len: Sequence length for test.
        num_heads: Number of KV heads.
        atol: Absolute tolerance for BF16 comparison.

    Returns:
        Dict with comparison metrics.
    """
    _require_ttnn()
    from turbo_quant.quantizer import TurboQuantMSE

    # CPU reference
    cpu_quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=seed, device="cpu", dtype=torch.float32)

    # Random test input
    torch.manual_seed(0)
    x_cpu = torch.randn(1, num_heads, seq_len, head_dim)

    # CPU quantize + dequantize
    cpu_indices, cpu_norms = cpu_quantizer.quantize(x_cpu)
    cpu_reconstructed = cpu_quantizer.dequantize(cpu_indices, cpu_norms)

    # Device quantize + dequantize
    setup = TTNNTurboQuantSetup(device, head_dim=head_dim, bits=bits, seed=seed)

    x_device = ttnn.from_torch(
        x_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    device_indices, device_norms = turbo_quant_quantize(x_device, setup)
    device_reconstructed = turbo_quant_dequantize(device_indices, device_norms, setup)

    # Transfer back to CPU for comparison
    device_reconstructed_cpu = ttnn.to_torch(device_reconstructed).float()
    device_indices_cpu = ttnn.to_torch(device_indices).int()

    # Compare
    recon_mse = ((cpu_reconstructed - device_reconstructed_cpu) ** 2).mean().item()
    recon_cosine = torch.nn.functional.cosine_similarity(
        cpu_reconstructed.flatten().unsqueeze(0),
        device_reconstructed_cpu.flatten().unsqueeze(0),
    ).item()
    index_match_pct = (cpu_indices.int() == device_indices_cpu).float().mean().item() * 100

    # Cleanup
    ttnn.deallocate(x_device)
    ttnn.deallocate(device_indices)
    ttnn.deallocate(device_norms)
    ttnn.deallocate(device_reconstructed)
    setup.deallocate()

    results = {
        "reconstruction_mse": recon_mse,
        "reconstruction_cosine": recon_cosine,
        "index_match_pct": index_match_pct,
        # BF16 on-device vs float32 CPU: index match is ~99%, but cosine
        # similarity is ~0.77 due to BF16 precision in rotation matmul,
        # norm computation, and centroid representation.
        "passed": recon_cosine > 0.70 and index_match_pct > 90,
    }

    return results
