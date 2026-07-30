# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Combined DRAM-core prefetch + consuming 1D matmul.

``ttnn.experimental.queue_tensor_prefetcher_request`` (fills a DRAM-sender
GlobalCircularBuffer over NOC, off the command queue) and the ``ttnn.linear``
that drains that GCB are always issued as a pair, against the *same* GCB and the
*same* 1D program config. As two separate calls the caller has to (a)
hand both the same ``global_cb``, (b) hand both the same ``program_config``, and
(c) pass a prefetch ``block_count`` that matches what the matmul expects -- three
couplings nothing enforces.

``prefetch_and_linear`` issues the pair from one call site so they cannot drift:
it derives ``block_count``, queues the request, then runs the consuming
``ttnn.linear`` with the same GCB and program config. Gather-in0 uses one block
per ring receiver. Mcast-in0 uses ``K_tiles / in0_block_w`` natural-order blocks
per receiver and requires a receiver-contiguous weight.

This is a host-side composition, not a device-level fusion: the prefetch still
runs on the DRAM-core (DRISC) path off the command queue while the matmul is
dispatched normally. Both halves act on the calling thread's current command
queue, and ``queue_id``/``cq_id`` here selects it for the pair at once: the
prefetch opts into capture (``capture_into_trace=True``) against that queue, and
the matmul dispatches on it, so a call inside a capture region captures -- and
replays -- both halves together. Taking the argument here rather than letting it
fall through ``**linear_kwargs`` is what keeps the pair together: ttnn's operation
decorator honours ``queue_id``/``cq_id`` on any op (it applies them by making that
queue current), so a passed-through ``cq_id`` would have steered the matmul alone.
"""

from contextlib import nullcontext

import ttnn


def prefetch_and_linear(
    input_tensor_a,
    weight,
    *,
    global_cb,
    program_config,
    queue_id=None,
    cq_id=None,
    **linear_kwargs,
):
    """Queue a DRAM-core prefetch of ``weight`` into ``global_cb``, then run the
    1D matmul (``ttnn.linear``) that consumes it.

    Gather-in0 preserves its existing batched/streaming behavior selected by
    ``program_config.stream_in1``. Mcast-in0 always uses natural FIFO order with
    ``stream_in1=False`` and can consume from a shallow GCB without a rotation table.

    Args:
        input_tensor_a: Activation (in0).
        weight: DRAM-sharded weight (in1) to prefetch and multiply by. Streaming gather
            and GCB-backed mcast require a receiver-contiguous weight layout.
        global_cb: DRAM-sender GlobalCircularBuffer shared by the prefetch and
            the matmul.
        program_config: 1D matmul program config driving the matmul.
        queue_id: Command queue for both halves -- the prefetch is captured against it
            (when it is recording a trace) and the matmul dispatches on it. Defaults to
            the calling thread's current queue, as set by ``ttnn.command_queue(n)``.
        cq_id: Alias for ``queue_id``, accepted because ttnn operations take either.
            ``queue_id`` wins if both are given, matching ttnn's own precedence.
        **linear_kwargs: Forwarded to ``ttnn.linear`` (e.g. ``memory_config``,
            ``compute_kernel_config``, ``dtype``, ``bias``).

    Returns:
        The ``ttnn.linear`` output tensor.
    """
    device = input_tensor_a.device()
    if program_config.mcast_in0 or program_config.stream_in1:
        block_count = ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(program_config, weight, global_cb)
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
    # queue_id wins over cq_id, as in ttnn's operation decorator. Making the queue current
    # for the whole block is what ties the two halves to it -- both read the thread's current
    # queue, so neither can end up on a different one.
    selected_cq_id = queue_id if queue_id is not None else cq_id
    with ttnn.command_queue(selected_cq_id) if selected_cq_id is not None else nullcontext():
        ttnn.experimental.queue_tensor_prefetcher_request(
            device,
            [request],
            global_cb=global_cb,
            # Let the prefetch be captured against the queue the matmul below dispatches on:
            # under trace capture both halves land in the one trace and replay together.
            # Leaving this False would capture the matmul but send the prefetch immediately,
            # so a replay would never refill the GCB and the matmul would hang.
            capture_into_trace=True,
        )
        return ttnn.linear(
            input_tensor_a,
            weight,
            program_config=program_config,
            global_cb=global_cb,
            **linear_kwargs,
        )
