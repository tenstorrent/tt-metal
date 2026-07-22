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
dispatched normally. The pairing does compose with trace capture -- pass the
recording CQ as ``cq_id`` and the request is captured (and replayed) alongside
the matmul.
"""

import ttnn


def prefetch_and_linear(
    input_tensor_a,
    weight,
    *,
    global_cb,
    program_config,
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
        cq_id: Command queue for the prefetch request. When that CQ is mid
            trace-capture the request is captured into the trace. Defaults to the
            current command queue.
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

    # Streaming gather needs identity ring rotation. Mcast consumes natural FIFO
    # order and therefore uses the rotation-free request.
    if program_config.stream_in1:
        request = (weight, block_count, list(range(block_count)))
    else:
        request = (weight, block_count)
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [request],
        global_cb=global_cb,
        cq_id=cq_id,
    )
    return ttnn.linear(
        input_tensor_a,
        weight,
        program_config=program_config,
        global_cb=global_cb,
        **linear_kwargs,
    )
