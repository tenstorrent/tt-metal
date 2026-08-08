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
per receiver.

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
    global_cb,
    program_config,
    **linear_kwargs,
):
    """Queue a DRAM-core prefetch of ``weight`` into ``global_cb``, then run the
    1D matmul (``ttnn.linear``) that consumes it.

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
            the matmul.
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
        global_cb=global_cb,
        # Capture against the queue the matmul below dispatches on, so both halves land in the
        # one trace. Left False, a capture region would take the matmul but send the prefetch
        # immediately -- a replay would never refill the GCB and the matmul would hang.
        capture_into_trace=True,
        cq_id=cq_id,
    )
    return ttnn.linear(
        input_tensor_a,
        weight,
        program_config=program_config,
        global_cb=global_cb,
        **linear_kwargs,
    )
