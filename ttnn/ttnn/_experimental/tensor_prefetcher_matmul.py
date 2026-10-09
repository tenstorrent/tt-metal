# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Combined DRAM-core prefetch + consuming 1D matmul.

``ttnn.experimental.queue_tensor_prefetcher_request`` (fills a DRAM-sender
delivery target over NOC, off the command queue) and the ``ttnn.linear``
that drains it are always issued as a pair, against the *same* target and the
*same* 1D program config. As two separate calls the caller has to (a)
hand both the same delivery target, (b) hand both the same ``program_config``, and
(c) pass a prefetch ``block_count`` that matches what the matmul expects -- three
couplings nothing enforces.

``prefetch_and_linear`` issues the pair from one call site so they cannot drift:
it derives ``block_count``, queues the request, then runs the consuming
``ttnn.linear`` with the same target and program config. Gather-in0 uses one block
per ring receiver. Mcast-in0 uses ``K_tiles / in0_block_w`` natural-order blocks
per receiver.

The delivery target is either a ``GlobalCircularBuffer`` or the list of ``PrefetcherPipe``
objects from one ``create_prefetcher_pipes_for_tensor_prefetcher`` call; for pipes the block count
comes from the program config rather than the ring.

This is a host-side composition, not a device-level fusion: the prefetch still
runs on the DRAM-core (DRISC) path off the command queue while the matmul is
dispatched normally. A ``queue_id``/``cq_id`` in ``**linear_kwargs`` still reaches
``ttnn.linear``, but is read out here as well so that it steers both halves: applied
to only the matmul it would leave the prefetch on whatever queue was already current,
and a capture region would then capture the halves apart.
"""

import ttnn


def prefetch_and_linear(
    input_tensor_a,
    weight,
    *,
    global_cb=None,
    prefetcher_pipes=None,
    program_config,
    **linear_kwargs,
):
    """Queue a DRAM-core prefetch of ``weight`` into ``global_cb`` or
    ``prefetcher_pipes``, then run the 1D matmul (``ttnn.linear``) that consumes it.

    Gather-in0 preserves its existing batched/streaming behavior selected by
    ``program_config.stream_in1``. Mcast-in0 always uses natural FIFO order with
    ``stream_in1=False`` and can consume from a shallow GCB without a rotation table,
    on either DRAM weight layout.

    Args:
        input_tensor_a: Activation (in0).
        weight: DRAM-sharded weight (in1) to prefetch and multiply by, in either the legacy
            WIDTH_SHARDED K-row-major or the receiver-contiguous layout. Streaming gather
            (``stream_in1``) requires the receiver-contiguous layout.
        global_cb: DRAM-sender GlobalCircularBuffer shared by the prefetch and
            the matmul. Supply exactly one of ``global_cb`` / ``prefetcher_pipes``.
        prefetcher_pipes: The DRAM-sender PrefetcherPipes (every pipe of one
            ``create_prefetcher_pipes_for_tensor_prefetcher`` call) shared by the prefetch and
            the matmul, as an alternative to ``global_cb``. Receiver-contiguous weight only;
            mcast-in0 or gather-in0.
        program_config: 1D matmul program config driving the matmul.
        **linear_kwargs: Forwarded to ``ttnn.linear`` (e.g. ``memory_config``,
            ``compute_kernel_config``, ``dtype``, ``bias``). A ``queue_id``/``cq_id``
            here steers both halves -- the prefetch is captured against that queue (when
            it is recording a trace) and the matmul dispatches on it -- and defaults to
            the calling thread's current queue, as set by ``ttnn.command_queue(n)``.
            Either spelling is accepted, as for any ttnn operation; ``queue_id`` wins if
            both are given, matching ttnn's own precedence.

    Returns:
        The ``ttnn.linear`` output tensor.
    """
    if (global_cb is None) == (prefetcher_pipes is None):
        raise ValueError("prefetch_and_linear needs exactly one delivery target: global_cb or prefetcher_pipes")
    # Passed through by name only when set: the bindings take a GCB or a (default-empty) pipe list,
    # neither of which accepts None for the other transport.
    target_kwargs = {"global_cb": global_cb} if global_cb is not None else {"prefetcher_pipes": prefetcher_pipes}

    device = input_tensor_a.device()
    if prefetcher_pipes is not None or program_config.mcast_in0 or program_config.stream_in1:
        block_count = ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(
            program_config, weight, **target_kwargs
        )
    else:
        # Gather consumes one K-block per ring position.
        block_count = global_cb.receiver_cores().num_cores()

    # Streaming gather needs identity ring rotation (``rotation[r] = r``). That table is
    # layout-agnostic -- identical for ROUND_ROBIN_1D and CONTIGUOUS_1D receiver-contiguous
    # weights -- because the kernel slices it by each weight's own global receiver position,
    # so no distribution-strategy argument is needed here. Mcast consumes natural FIFO order
    # and therefore uses the rotation-free request.
    if program_config.stream_in1:
        request = (weight, block_count, list(range(block_count)))
    else:
        request = (weight, block_count)
    # Read (not popped) out of the kwargs the way ttnn's own operation wrapper resolves it
    # -- FastOperation.__call__ in ttnn/decorators.py picks on the keyword being present, not
    # on its value, so an explicit queue_id=None wins over a cq_id rather than falling through
    # to it. Only the prefetch half needs it named here; the keyword itself stays in
    # linear_kwargs and reaches ttnn.linear as the caller wrote it.
    cq_id = None
    if "queue_id" in linear_kwargs:
        cq_id = linear_kwargs["queue_id"]
    elif "cq_id" in linear_kwargs:
        cq_id = linear_kwargs["cq_id"]
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [request],
        **target_kwargs,
        # Capture against the queue the matmul below dispatches on, so both halves land in the
        # one trace. Left False, a capture region would take the matmul but send the prefetch
        # immediately -- a replay would never refill the ring and the matmul would hang.
        capture_into_trace=True,
        cq_id=cq_id,
    )
    return ttnn.linear(
        input_tensor_a,
        weight,
        program_config=program_config,
        **target_kwargs,
        **linear_kwargs,
    )


def prefetch_and_sparse_matmul(
    input_tensor_a,
    weight,
    sparsity,
    *,
    prefetcher_pipes,
    program_config,
    signal_id=None,
    **sparse_matmul_kwargs,
):
    """Queue a DRAM-core prefetch of the experts ``sparsity`` selects from ``weight`` into
    ``prefetcher_pipes``, then run the ``ttnn.sparse_matmul`` that consumes them.

    The prefetcher reads ``sparsity`` on device when it reaches the request and streams, in ascending
    expert order, only the experts whose entry is non-zero -- the order ``ttnn.sparse_matmul`` scans
    the same mask in. Because the selection happens on device, a trace captured around this call
    replays correctly when the mask changes between replays.

    Args:
        input_tensor_a: Activation (in0), interleaved: ``[1, 1, M, K]`` to broadcast over the
            experts, or ``[1, E, M, K]`` with ``is_input_a_sparse=True``.
        weight: The fused ``[1, E, K, N]`` expert weight, receiver-contiguous with an NdShardSpec
            shard of ``[E, K, N / ring_size]`` so every receiver slab holds all E experts.
        sparsity: ``[1, 1, 1, E]`` ROW_MAJOR mask, one page. It must be in place when the
            prefetcher reaches the request and must not change until the matmul has run, or the
            pipes deadlock. Fence a host write with ``ttnn.experimental.wait_for_cq_on_tensor_prefetcher``
            before this call; for a mask an op writes on device, pass ``signal_id``.
        prefetcher_pipes: Every pipe of one ``create_prefetcher_pipes_for_tensor_prefetcher`` call,
            whose receivers are the matmul's workers.
        program_config: ``MatmulMultiCoreReuseMultiCast1DProgramConfig`` with ``mcast_in0=True``
            and one output block per worker.
        signal_id: When set, the prefetcher reads ``sparsity`` only after every op already enqueued
            on the matmul's queue has run: this call raises op signal ``signal_id`` on that queue
            (``ttnn.experimental.signal_tensor_prefetcher``) and queues a wait on it ahead of the
            request. Both are captured into a trace with the rest, so a mask written by an op inside
            the same trace is read after that op on every replay. Every wait on the signal must pair
            with one raise of it, so don't raise it elsewhere.
        **sparse_matmul_kwargs: Forwarded to ``ttnn.sparse_matmul`` (e.g. ``nnz``,
            ``is_input_a_sparse``, ``memory_config``, ``compute_kernel_config``, ``dtype``). Prefer
            leaving ``nnz`` unset: an ``nnz`` that differs from ``count_nonzero(sparsity)`` stalls the
            prefetcher as well as the matmul. ``queue_id``/``cq_id`` steer both halves, as in
            ``prefetch_and_linear``.

    Returns:
        The ``ttnn.sparse_matmul`` output tensor.
    """
    device = input_tensor_a.device()
    block_count = ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(
        program_config, weight, prefetcher_pipes=prefetcher_pipes
    )
    # Resolved as prefetch_and_linear resolves it, so the request is captured on the queue the
    # matmul dispatches on.
    cq_id = None
    if "queue_id" in sparse_matmul_kwargs:
        cq_id = sparse_matmul_kwargs["queue_id"]
    elif "cq_id" in sparse_matmul_kwargs:
        cq_id = sparse_matmul_kwargs["cq_id"]
    if signal_id is not None:
        # The signal runs on the queue after the op that wrote the mask, and the wait holds the request
        # until it has.
        ttnn.experimental.signal_tensor_prefetcher(device, signal_id, cq_id=cq_id)
        ttnn.experimental.queue_tensor_prefetcher_wait_for_signal(
            device, signal_id, capture_into_trace=True, cq_id=cq_id
        )
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        # Rotation-free: mcast_in0 consumes each expert's K-blocks in natural FIFO order.
        [(weight, block_count, [], sparsity)],
        prefetcher_pipes=prefetcher_pipes,
        capture_into_trace=True,
        cq_id=cq_id,
    )
    return ttnn.sparse_matmul(
        input_tensor_a,
        weight,
        sparsity=sparsity,
        program_config=program_config,
        prefetcher_pipes=prefetcher_pipes,
        **sparse_matmul_kwargs,
    )
