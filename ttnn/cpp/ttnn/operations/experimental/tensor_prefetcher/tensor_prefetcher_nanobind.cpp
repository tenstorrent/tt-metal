// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_prefetcher_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "tensor_prefetcher.hpp"
#include "ttnn/global_circular_buffer.hpp"

namespace ttnn::operations::experimental {

namespace {

// Bound in place of create_prefetcher_pipes_for_tensor_prefetcher so that each returned pipe -- not
// the list holding them -- keeps the device alive. A caller may keep one pipe and drop the list, and
// a pipe's destructor reaches into the device to free its L1; nb::keep_alive<0, 1> would tie only
// the list's own lifetime to the device (and a list cannot be a keep-alive nurse at all).
nb::list create_prefetcher_pipes_for_tensor_prefetcher_py(
    const nb::object& space_object,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool support_multi_receiver_shards) {
    // ttnn::bind_function calls this with the GIL released, and everything below touches Python
    // objects.
    nb::gil_scoped_acquire gil;
    const auto pipes = create_prefetcher_pipes_for_tensor_prefetcher(
        nb::cast<tt::tt_metal::experimental::PrefetcherPipeSpace&>(space_object),
        bank_to_receivers,
        support_multi_receiver_shards);

    nb::list pipe_list;
    for (const auto& pipe : pipes) {
        nb::object pipe_object = nb::cast(pipe);
        nb::detail::keep_alive(pipe_object.ptr(), space_object.ptr());
        pipe_list.append(pipe_object);
    }
    return pipe_list;
}

}  // namespace

void bind_tensor_prefetcher(nb::module_& mod) {
    ttnn::bind_function<"is_tensor_prefetcher_supported", "ttnn.experimental.">(
        mod,
        R"doc(
            Return True if the Tensor prefetcher (DRISC) is supported on `mesh_device`,
            i.e. programmable DRAM cores are available (Blackhole with firmware >= 19.12.0.0).
            When this returns False,
            start_tensor_prefetcher would raise, so callers can use this to skip instead.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device to query.

            Returns:
                bool
        )doc",
        &is_tensor_prefetcher_supported,
        nb::arg("mesh_device"));

    ttnn::bind_function<"start_tensor_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Start the queueable Tensor prefetcher (DRISC) on `mesh_device`. Returns
            immediately; one DRISC kernel parks on a per-(device, sender-core) H2D socket
            waiting for requests. Pair with queue_tensor_prefetcher_request and
            stop_tensor_prefetcher.

            Only one Tensor prefetcher may be active per mesh device at a time.
            Receiver count is per pipe (read from each pipe's sender config page on every
            request), so a single prefetcher can serve pipe sets with different receiver
            counts.

            On Blackhole, the GDDR Memory Controller Multi-Port Front End (MPFE) uses
            weighted round-robin arbitration. Weights are in [0, 7], and higher relative
            values receive more service. The hardware slots P1/P2/P3 correspond to DRAM
            tiles D0/D1/D2; the arguments below name traffic roles, which Metal maps to
            each bank's D tiles using its worker_endpoint assignment.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device to launch on.
                free_sender_mpfe_weight (Optional[int]): active free-sender MPFE weight.
                    None uses the Metal default.
                noc1_sender_mpfe_weight (Optional[int]): active NOC1-sender MPFE weight.
                    None uses the Metal default.
                ordinary_mpfe_weight (Optional[int]): ordinary-operation MPFE weight.
                    None uses the Metal default.
                dynamic_mpfe_weighting (Optional[bool]): when True, idle both prefetch senders
                    at the ordinary weight and lower each private sender slot only while it
                    handles a request. None uses the Metal default.

            The benchmark tuning script enables TT_METAL_BENCHMARK_TENSOR_PREFETCHER_ENABLE
            and supplies per-field environment overrides. Those benchmark-only values take
            precedence over these keyword arguments while the enable variable is set.

            Two sender kernels are provisioned per DRAM bank. Each queued pipe set selects
            one or both senders per bank; unused senders remain parked on their sockets.
        )doc",
        &start_tensor_prefetcher,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("free_sender_mpfe_weight") = std::nullopt,
        nb::arg("noc1_sender_mpfe_weight") = std::nullopt,
        nb::arg("ordinary_mpfe_weight") = std::nullopt,
        nb::arg("dynamic_mpfe_weighting") = std::nullopt);

    ttnn::bind_function<"queue_tensor_prefetcher_request", "ttnn.experimental.">(
        mod,
        R"doc(
            Queue one prefetch request. Non-blocking. Each pipe's write cursors are
            preserved across requests, so successive Queue calls against the same pipes
            resume where the previous call left off. Successive Queue calls can target
            different pipe sets.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device whose prefetcher to queue on.
                tensors (List[Tuple[ttnn.Tensor, int] | Tuple[ttnn.Tensor, int, List[int]] |
                    Tuple[ttnn.Tensor, int, List[int], ttnn.Tensor]]): the full, flattened list of
                    weights to prefetch (at least one), streamed in list order. Each item is
                    (weight, block_count), (weight, block_count, rotation) to enable per-tensor
                    streaming, or (weight, block_count, rotation, selector) for a grouped weight
                    (see selector below; pass rotation=[] for batched delivery). block_count is the number of
                    K-blocks to divide that tensor's K dimension into (the consumer matmul
                    waits on block_count pages per layer). Pass distinct tensors for distinct
                    layers, or repeat a tensor to replay it.

                    rotation (receiver-contiguous layout only; omit/empty == batched) is the
                    per-receiver streaming ring-rotation table, indexed by global ring position
                    and of length total_receivers (== ring_size == block_count), each entry in
                    [0, block_count). It makes the kernel deliver that tensor's K-blocks in the
                    host-specified ring-rotated order so the consuming matmul can stream them FIFO
                    (and start before the whole tensor lands, allowing a shallow ring). rotation[r]
                    = r reproduces the natural topology order; the matmul must consume in the
                    matching order, else it deadlocks.

                    selector (grouped weights only) is a mask with one entry per group, for a
                    weight whose NdShardSpec shard is (num_groups, K, N / ring_size) -- e.g. a fused
                    [1, E, K, N] expert stack. The prefetcher reads it on device and streams, in
                    ascending group order, only the groups whose mask entry is non-zero, which is
                    what ttnn.sparse_matmul consumes for the same mask. ROW_MAJOR BFLOAT16 or UINT16
                    of shape [..., num_groups] (num_groups <= 256), DRAM interleaved or on one L1
                    core. The mask must be written before this request is queued (fence the writing
                    queue with wait_for_cq_on_tensor_prefetcher) and must not change until the
                    consuming sparse_matmul has run, else the prefetcher and the matmul disagree on
                    which experts arrive and deadlock. Reading it on device is what keeps a
                    captured request correct when the mask changes between trace replays.
                    wait_for_cq_on_tensor_prefetcher cannot be captured into a trace, so a mask
                    written by an op inside the same trace is not fenced on replay: the mask must
                    be written, and fenced, before execute_trace. A grouped weight without a
                    selector streams every group.
                prefetcher_pipes (List[PrefetcherPipe]): the DRAM-sender PrefetcherPipes (created via
                    ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher) to deliver into.
                    Pass every pipe from one such call, in any order; a subset or a mix of two calls'
                    pipes is rejected. A legacy K-row-major tensor needs one pipe per bank
                    (support_multi_receiver_shards=True). A tensor's per-receiver block size need not
                    equal the pipes' entry_size nor divide the ring, so size the ring for the
                    consumer: one block is enough for the transport, two for a consumer that keeps a
                    block of lookahead.
                device_subset (Optional[MeshCoordinateRangeSet]): subset of the mesh that
                    processes this request. Defaults to the full mesh.
                capture_into_trace (bool): whether this request may be captured into a trace.
                    When True and the current command queue is mid trace-capture, the request
                    is captured into the trace instead of being sent immediately, and is
                    re-sent on every execute_trace of that trace. Defaults to False: the
                    request is always sent immediately and is never captured, whatever any
                    command queue is doing. Which queue counts as current follows the usual
                    ttnn convention — pass cq_id=n, or wrap the call in ttnn.command_queue(n).

            Returns:
                None
        )doc",
        &queue_tensor_prefetcher_request,
        nb::arg("mesh_device"),
        nb::arg("tensors"),
        nb::kw_only(),
        nb::arg("prefetcher_pipes"),
        nb::arg("device_subset") = nb::none(),
        nb::arg("capture_into_trace") = false);

    ttnn::bind_function<"wait_for_cq_on_tensor_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Fence the Tensor prefetcher against work enqueued on a command queue.
            Every prefetch request queued after this call waits until all work previously
            enqueued on `cq_id` has completed on device before the prefetcher reads DRAM.
            Use this to guarantee data written over `cq_id` has landed before the
            prefetcher streams it.

            Call synchronously on the host thread that issued the data writes — after those
            writes, and before the queue_tensor_prefetcher_request that consumes them.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device whose prefetcher to fence.
                cq_id (Optional[int]): the command queue to fence against. Defaults to the
                    calling thread's current queue (also what ttnn.command_queue(n) sets).
                device_subset (Optional[MeshCoordinateRangeSet]): subset of the mesh to
                    fence. Defaults to the full mesh.

            Returns:
                None
        )doc",
        &wait_for_cq_on_tensor_prefetcher,
        nb::arg("mesh_device"),
        nb::arg("cq_id") = nb::none(),
        nb::kw_only(),
        nb::arg("device_subset") = nb::none());

    ttnn::bind_function<"stop_tensor_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Push the stop sentinel to every socket, join the host worker thread, and wait
            for the kernels to exit. No-op if no prefetcher is active.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device whose prefetcher to stop.
        )doc",
        &stop_tensor_prefetcher,
        nb::arg("mesh_device"));

    ttnn::bind_function<"create_prefetcher_pipes_for_tensor_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Create the PrefetcherPipes whose senders are programmable DRAM cores (Blackhole DRISCs),
            the Tensor prefetcher's delivery target. Each bank id is mapped to its DRAM sender
            cores; receiver sets across banks must be disjoint.

            Returns one PrefetcherPipe per DRAM sender core, bank-major: a bank's pipes are
            adjacent, and the leading one owns that bank's leading receivers. Each pipe carries the
            bank-local slab base its sender owns, so queueing accepts them in any order. It
            requires all of them, and rejects a subset or a mix of two calls' pipes.

            Consumers bind the pipes through ProgramRunArgs and read them through the device-side
            PrefetcherPipe (wait_front / scoped_read_lock / pop_front). Keep the pipes alive for as
            long as any program uses them: dropping the last reference to one frees its ring and
            config.

            Args:
                space: Existing PrefetcherPipeSpace that owns the ring geometry and reserves
                    enough DRAM sender capacity. Keep it alive while using the returned pipes.
                bank_to_receivers: List of (bank_id, receivers) pairs.
                support_multi_receiver_shards: If True, a bank's shard may feed multiple receivers,
                    which forces a single sender per bank. Defaults to False (receiver-contiguous),
                    letting a bank with two or more receivers split them across two DRISC senders.
        )doc",
        &create_prefetcher_pipes_for_tensor_prefetcher_py,
        nb::arg("space"),
        nb::arg("bank_to_receivers"),
        nb::arg("support_multi_receiver_shards") = false);

    ttnn::bind_function<"tensor_prefetcher_block_count_for_matmul_1d", "ttnn.experimental.">(
        mod,
        R"doc(
            Compute and validate the block_count to pair with a DRAM weight in
            queue_tensor_prefetcher_request for a gather_in0 or mcast_in0 1D matmul. Gather returns
            the receiver/ring count. Mcast returns weight_K_tiles / in0_block_w and uses natural
            FIFO order.

            Args:
                program_config: The 1D matmul program config that will consume the weight.
                weight: The DRAM weight tensor, in either layout. A legacy K-row-major weight takes
                    its receivers per bank from the pipes, which need one per bank.
                prefetcher_pipes: The DRAM-sender PrefetcherPipes the prefetcher and matmul share.

            Returns:
                int: the validated block_count.
        )doc",
        static_cast<uint32_t (*)(
            const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig&,
            const ttnn::Tensor&,
            const ttnn::PrefetcherPipeList&)>(
            &ttnn::global_circular_buffer::tensor_prefetcher_block_count_for_matmul_1d),
        nb::arg("program_config"),
        nb::arg("weight"),
        nb::arg("prefetcher_pipes"));
}

}  // namespace ttnn::operations::experimental
