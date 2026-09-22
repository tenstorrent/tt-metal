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
drift. The prefetch request's ``block_count`` and the matmul's
``global_cb_k_blocks`` are the same number -- how many GCB pages one receiver's
slab is cut into -- and a disagreement is a device hang rather than an error, so
neither is exposed separately: pass ``k_blocks`` once and both sides get it.

``k_blocks=1`` (the default) makes a page a whole slab, so the GCB has to hold at
least one slab per receiver. Higher values cut the slab into that many equal
blocks of K-rows, which the matmul streams and accumulates over. That is what
lets a GCB be smaller than a slab, and -- because the page size is then a
property of the page rather than of the weight -- what lets several weights with
differently sized slabs share one GCB. Use ``matmul_decode_k_blocks`` to derive
``k_blocks`` from the page size you want.

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

# Default GCB depth in pages. Two is the minimum a streamed (k_blocks > 1) weight can run on
# -- the matmul holds one page un-acked while the next is delivered -- and with the default
# k_blocks=1 it is two whole slabs, which lets the prefetcher land the next invocation's
# weights while the current matmul is still computing.
_DEFAULT_NUM_PAGES = 2


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


def _slab_k_tiles(weight, slab_h):
    """Rows of one slab, in tiles -- the dimension a GCB page is cut along."""
    return slab_h // weight.tile.tile_shape[0]


def matmul_decode_k_blocks(weight, num_receivers, page_bytes, *, slab_shape=None):
    """How many GCB pages of ``page_bytes`` one receiver's weight slab is cut into.

    This is the number to hand to ``make_matmul_decode_gcb``, ``prefetch_and_matmul_decode``
    and (if you issue the pair yourself) both the prefetch request's ``block_count`` and
    ``matmul_decode``'s ``global_cb_k_blocks``.

    Deriving it from a page size rather than picking it per weight is the point: weights that
    agree on ``page_bytes`` can share one GCB however much their slabs differ in size, since
    the GCB is sized and credited in pages.

    Args:
        weight: the DRAM ND-sharded weight. Only its shape/dtype are read.
        num_receivers: GCB receiver count, used to derive the full width-sharded slab.
        page_bytes: the wanted page size. Must divide the slab into whole pages of whole
            K-rows.
        slab_shape: per-receiver slab ``[height, width]`` in elements, as for
            ``make_matmul_decode_gcb``. Defaults to full width-sharded.

    Returns:
        The page count, at least 1.
    """
    slab_h, slab_w = _resolve_slab_shape(weight, num_receivers, slab_shape)
    slab_bytes = _slab_bytes(weight, slab_h, slab_w)
    if page_bytes <= 0 or slab_bytes % page_bytes != 0:
        raise ValueError(
            f"a {page_bytes} B page does not divide this weight's {slab_bytes} B slab "
            f"([{slab_h}, {slab_w}] of {weight.dtype}) into whole pages"
        )
    k_blocks = slab_bytes // page_bytes
    # A page is a block of whole K-rows of the slab, so an even split in bytes is not enough:
    # the rows have to split evenly too. They can disagree -- a 2-tile-tall slab has a byte
    # size divisible by 4 -- and the mismatch would only surface as a rejected matmul.
    k_tiles = _slab_k_tiles(weight, slab_h)
    if k_tiles % k_blocks != 0:
        raise ValueError(
            f"a {page_bytes} B page cuts this weight's slab into {k_blocks} pages, but its "
            f"{k_tiles} tile-rows do not divide into that many blocks of whole rows"
        )
    return k_blocks


def make_matmul_decode_gcb(
    device, weight, bank_to_receivers, *, slab_shape=None, k_blocks=1, num_pages=_DEFAULT_NUM_PAGES
):
    """Build a DRAM-sender GCB sized to hold ``num_pages`` weight pages per receiver.

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
        k_blocks: how many pages one slab is cut into, from ``matmul_decode_k_blocks``.
            1 (the default) makes a page a whole slab.
        num_pages: GCB depth in pages. 2 (the default) is one slab of run-ahead when
            ``k_blocks`` is 1, and the minimum a streamed weight can run on otherwise --
            the matmul holds one page un-acked while the next is delivered. Deeper rings
            buy the prefetcher more run-ahead.

    Returns:
        A ``ttnn.GlobalCircularBuffer`` to pass as ``global_cb`` to both the prefetch
        request and ``matmul_decode``.
    """
    if num_pages < 1:
        raise ValueError(f"num_pages must be >= 1, got {num_pages}")
    if k_blocks > 1 and num_pages < 2:
        raise ValueError(f"streaming a slab as {k_blocks} pages needs a GCB of at least 2 pages, got {num_pages}")
    num_receivers = sum(crs.num_cores() for _, crs in bank_to_receivers)
    slab_h, slab_w = _resolve_slab_shape(weight, num_receivers, slab_shape)
    _check_slab_tiles_the_weight(weight, num_receivers, slab_h, slab_w)
    slab_bytes = _slab_bytes(weight, slab_h, slab_w)
    if slab_bytes % k_blocks != 0 or _slab_k_tiles(weight, slab_h) % k_blocks != 0:
        raise ValueError(f"a [{slab_h}, {slab_w}] slab does not split into {k_blocks} pages of whole K-rows")
    size = num_pages * (slab_bytes // k_blocks)
    return ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, size)


def prefetch_and_matmul_decode(
    input_tensor_a,
    weight,
    *,
    global_cb,
    k_blocks=1,
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
        k_blocks: how many GCB pages one receiver's slab is cut into. Must be the value the
            GCB was sized with; derive it with ``matmul_decode_k_blocks``. Both the prefetch
            and the matmul are given it here, which is the coupling this function exists for.
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
        [(weight, k_blocks)],
        global_cb=global_cb,
        cq_id=cq_id,
    )
    return ttnn.experimental.matmul_decode(
        input_tensor_a,
        weight,
        global_cb=global_cb,
        global_cb_k_blocks=k_blocks,
        **matmul_kwargs,
    )
