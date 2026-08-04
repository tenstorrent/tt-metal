# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Combined DRAM-core prefetch + consuming ``matmul_decode``.

``ttnn.experimental.queue_tensor_prefetcher_request`` (fills a DRAM-sender
GlobalCircularBuffer over NOC, off the command queue) and the
``ttnn.experimental.matmul_decode`` that drains that GCB are always issued as a
pair, against the *same* GCB. As two separate calls the caller has to (a) hand
both the same ``global_cb`` and (b) pass a prefetch ``block_count`` that matches
what the matmul expects -- two couplings nothing enforces.

``prefetch_and_matmul_decode`` issues the pair from one call site so they cannot
drift. ``block_count`` is always 1, in every ``matmul_decode`` mode: each
weight-holding core owns exactly one contiguous slab and its compute kernel
indexes in1 tiles by absolute position within that slab, so the whole slab must
be resident for the duration and therefore has to arrive as a single GCB page.

The per-receiver slab, and the order the slabs must appear in, differ per mode:

===================== ===================== ================ =============================================
Mode                  Receivers             Slab shape       Slab ``idx`` -> weight block
===================== ===================== ================ =============================================
Full width-sharded    ``N_blocks``          ``[K, N/N_blocks]``  ``n_idx = idx``
Partial width-sharded ``K_blocks*N_blocks`` ``[Kc, Nc]``     ``k_idx = idx // N_blocks``, ``n_idx = idx % N_blocks``
Batched width-sharded ``b_blocks*n_blocks`` ``[Bc*K, Nc]``   ``b_idx = idx // n_blocks``, ``n_idx = idx % n_blocks``
===================== ===================== ================ =============================================

N is the fast-varying dimension in both two-dimensional modes. The slab shape is
in every case the weight's ND shard shape, and slab ``idx`` is its ND shard
index, so laying the weight out correctly is a matter of choosing the ND shard
shape -- but the *ring* order must agree with it independently, and a mismatch
is silently wrong results rather than an error.

This is a host-side composition, not a device-level fusion: the prefetch runs on
the DRAM-core (DRISC) path off the command queue while the matmul is dispatched
normally. The pairing composes with trace capture -- pass the recording CQ as
``cq_id`` and the request is captured (and replayed) alongside the matmul.
"""

import ttnn

# One GCB page per receiver per invocation, carrying that receiver's whole weight slab.
_BLOCK_COUNT = 1

# Default GCB depth in slabs. Two lets the prefetcher land the next invocation's weights
# while the current matmul is still computing; one serializes them.
_DEFAULT_NUM_SLABS = 2


def _resolve_slab_shape(weight, num_receivers, slab_shape):
    """The per-receiver ``(height, width)`` slab in elements, i.e. the weight's ND shard shape:
    ``[Kc, Nc]`` for partial width-sharded, ``[Bc*K, Nc]`` for batched. ``slab_shape`` may be
    ``None`` for full width-sharded, where it is ``[K, N/num_receivers]``.
    """
    if slab_shape is None:
        n = weight.shape[-1]
        if n % num_receivers != 0:
            raise ValueError(f"weight N={n} must be divisible by the receiver count {num_receivers}")
        return int(weight.shape[-2]), int(n // num_receivers)
    return int(slab_shape[0]), int(slab_shape[1])


def _check_slab_tiles_the_weight(weight, num_receivers, slab_h, slab_w):
    """Reject a slab shape / receiver count that does not cut the weight into one slab per receiver.

    This mirrors the shard count the device-side validator compares against the GCB receiver
    count, but reaches the caller before a GCB is even built. Without it, an over- or
    under-counted receiver set is only discovered as slabs nobody sends, i.e. a hang.
    """
    shape = [int(d) for d in weight.shape]
    num_shards = 1
    for dim in shape[:-2]:
        num_shards *= dim
    num_shards *= -(-shape[-2] // slab_h) * -(-shape[-1] // slab_w)
    if num_shards != num_receivers:
        raise ValueError(
            f"a [{slab_h}, {slab_w}] slab cuts weight {shape} into {num_shards} shards, "
            f"but the GCB has {num_receivers} receivers"
        )


def _slab_bytes(weight, slab_h, slab_w):
    """Byte size of one receiver's weight slab, in whole tiles."""
    # Verified against this build: ttnn.Tensor exposes `.tile` and `.dtype`; there is no
    # `.tensor_spec` and no `ttnn.datatype_to_dataformat_converter` in Python.
    # `tile.get_tile_size(dtype)` returns 2048 for bfloat16, 1088 for bfloat8_b, 576 for bfloat4_b.
    tile = weight.tile
    tile_bytes = tile.get_tile_size(weight.dtype)
    tile_h, tile_w = tile.tile_shape[0], tile.tile_shape[1]
    if slab_h % tile_h != 0 or slab_w % tile_w != 0:
        # Unchecked, the divisions below truncate and silently undersize the GCB -- a page
        # then arrives short and the matmul's reader hangs waiting for bytes that were
        # never queued, rather than raising a clean error here.
        raise ValueError(f"weight slab [{slab_h}, {slab_w}] must be tile-aligned (tile is {tile_h}x{tile_w})")
    return (slab_h // tile_h) * (slab_w // tile_w) * tile_bytes


def make_matmul_decode_gcb(device, weight, bank_to_receivers, *, slab_shape=None, num_slabs=_DEFAULT_NUM_SLABS):
    """Build a DRAM-sender GCB sized to hold ``num_slabs`` weight slabs per receiver.

    Args:
        device: the mesh device.
        weight: the DRAM ND-sharded (receiver-contiguous) weight this GCB will carry.
            Only its shape/dtype are read here.
        bank_to_receivers: list of ``(dram_bank_id, ttnn.CoreRangeSet)`` pairs. The
            receiver at ring position ``p`` must be the B core whose row-major index in
            the matmul's B grid is ``p``, and the weight's shard ``p`` must be the block
            that core computes -- for full width-sharded its N-column block, for the two
            two-dimensional modes the block at ``(p // N_blocks, p % N_blocks)`` resp.
            ``(p // n_blocks, p % n_blocks)`` (see the module docstring's table; N is the
            fast-varying dimension). Build it with ``bank_receivers_strided`` for
            ``ROUND_ROBIN_1D`` weights or ``bank_receivers_contiguous`` for
            ``CONTIGUOUS_1D``.
        slab_shape: the per-receiver slab as ``[height, width]`` in elements -- ``[Kc, Nc]``
            for partial width-sharded, ``[Bc*K, Nc]`` for batched. Defaults to the full
            width-sharded ``[K, N/num_receivers]``. It must cut ``weight`` into exactly one
            shard per receiver, which is checked here.
        num_slabs: GCB depth in whole slabs. Must be at least 1; 2 (the default) lets the
            prefetch of the next invocation overlap the current matmul.

    Returns:
        A ``ttnn.GlobalCircularBuffer`` to pass as ``global_cb`` to both the prefetch
        request and ``matmul_decode``.
    """
    if num_slabs < 1:
        raise ValueError(f"num_slabs must be >= 1, got {num_slabs}")
    num_receivers = sum(crs.num_cores() for _, crs in bank_to_receivers)
    slab_h, slab_w = _resolve_slab_shape(weight, num_receivers, slab_shape)
    _check_slab_tiles_the_weight(weight, num_receivers, slab_h, slab_w)
    size = num_slabs * _slab_bytes(weight, slab_h, slab_w)
    return ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, size)


def prefetch_and_matmul_decode(
    input_tensor_a,
    weight,
    *,
    global_cb,
    cq_id=None,
    **matmul_kwargs,
):
    """Queue a DRAM-core prefetch of ``weight`` into ``global_cb``, then run the
    ``matmul_decode`` that consumes it.

    Args:
        input_tensor_a: activation (in0), L1 width-sharded along K.
        weight: DRAM ND-sharded (receiver-contiguous) weight (in1), one slab per GCB
            receiver in the mode's receiver order (see the module docstring's table).
        global_cb: DRAM-sender GlobalCircularBuffer shared by the prefetch and the matmul.
            Build it with ``make_matmul_decode_gcb`` rather than passing an arbitrarily
            sized one. A GCB too small for a single slab per receiver is rejected on the
            host by every ``matmul_decode`` program factory, but a GCB paired with a
            prefetch request whose ``block_count`` or tensor disagrees with what the matmul
            waits for is a device hang, not a clean error -- which is what issuing the pair
            from here prevents.
        cq_id: command queue for the prefetch request. When that CQ is mid trace-capture
            the request is captured into the trace. Defaults to the current command queue.
        **matmul_kwargs: forwarded to ``ttnn.experimental.matmul_decode``
            (e.g. ``dtype``, ``output_mem_config``).

    Returns:
        The ``matmul_decode`` output tensor.
    """
    device = input_tensor_a.device()
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [(weight, _BLOCK_COUNT)],
        global_cb=global_cb,
        cq_id=cq_id,
    )
    return ttnn.experimental.matmul_decode(
        input_tensor_a,
        weight,
        global_cb=global_cb,
        **matmul_kwargs,
    )
