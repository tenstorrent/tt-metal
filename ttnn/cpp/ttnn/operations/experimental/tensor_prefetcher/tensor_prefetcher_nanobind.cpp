// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_prefetcher_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
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
    const nb::object& mesh_device_object,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::tt_metal::BufferType buffer_type,
    bool support_multi_receiver_shards) {
    // ttnn::bind_function calls this with the GIL released, and everything below touches Python
    // objects.
    nb::gil_scoped_acquire gil;
    const auto pipes = create_prefetcher_pipes_for_tensor_prefetcher(
        nb::cast<tt::tt_metal::distributed::MeshDevice*>(mesh_device_object),
        bank_to_receivers,
        entry_size,
        num_entries,
        buffer_type,
        support_multi_receiver_shards);

    nb::list pipe_list;
    for (const auto& pipe : pipes) {
        nb::object pipe_object = nb::cast(pipe);
        nb::detail::keep_alive(pipe_object.ptr(), mesh_device_object.ptr());
        pipe_list.append(pipe_object);
    }
    return pipe_list;
}

}  // namespace

void bind_tensor_prefetcher(nb::module_& mod) {
    // One durable ring per sender, read-only from Python: a delivery target is a list of these, and
    // a caller names one to ask which core sends it, which cores read it, and how big its ring is.
    // Not constructible here -- the public constructor builds a *worker*-sender pipe, which the
    // prefetcher rejects; the DRAM-sender ones come from
    // create_prefetcher_pipes_for_tensor_prefetcher.
    nb::class_<tt::tt_metal::experimental::PrefetcherPipe>(mod, "PrefetcherPipe")
        .def("config_address", &tt::tt_metal::experimental::PrefetcherPipe::config_address)
        .def("buffer_address", &tt::tt_metal::experimental::PrefetcherPipe::buffer_address)
        .def("ring_size", &tt::tt_metal::experimental::PrefetcherPipe::ring_size)
        .def("initial_entry_size", &tt::tt_metal::experimental::PrefetcherPipe::initial_entry_size)
        // DRAM-logical for a DRAM sender, so its x is the bank id this pipe is fed from.
        .def("sender_core", &tt::tt_metal::experimental::PrefetcherPipe::sender_core)
        .def(
            "receiver_cores",
            &tt::tt_metal::experimental::PrefetcherPipe::receiver_cores,
            nb::rv_policy::reference_internal)
        .def("sender_core_type", [](const tt::tt_metal::experimental::PrefetcherPipe& pipe) {
            return pipe.sender_core_type() == tt::tt_metal::experimental::SenderCoreType::Dram ? "dram" : "worker";
        });

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
            Receiver count is per-GCB (read from each GCB's sender state block on every
            request), so a single prefetcher can serve GCBs with different num_receivers
            values.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device to launch on.

            Two sender kernels are provisioned per DRAM bank. Each queued GCB selects one
            or both senders per bank; unused senders remain parked on their sockets.
        )doc",
        &start_tensor_prefetcher,
        nb::arg("mesh_device"));

    ttnn::bind_function<"queue_tensor_prefetcher_request", "ttnn.experimental.">(
        mod,
        R"doc(
            Queue one prefetch request. Non-blocking. Per-GCB ring-buffer state is
            preserved across requests, so successive Queue calls against the same GCB
            resume where the previous call left off. Successive Queue calls can target
            different GCBs.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device whose prefetcher to queue on.
                tensors (List[Tuple[ttnn.Tensor, int] | Tuple[ttnn.Tensor, int, List[int]]]): the
                    full, flattened list of weights to prefetch (at least one), streamed in
                    list order. Each item is (weight, block_count) or, to enable per-tensor
                    streaming, (weight, block_count, rotation). block_count is the number of
                    K-blocks to divide that tensor's K dimension into (the consumer matmul
                    waits on block_count pages per layer). Pass distinct tensors for distinct
                    layers, or repeat a tensor to replay it.

                    rotation (receiver-contiguous layout only; omit/empty == batched) is the
                    per-receiver streaming ring-rotation table, indexed by global ring position
                    and of length total_receivers (== ring_size == block_count), each entry in
                    [0, block_count). It makes the kernel deliver that tensor's K-blocks in the
                    host-specified ring-rotated order so the consuming matmul can stream them FIFO
                    (and start before the whole tensor lands, allowing a shallow GCB). rotation[r]
                    = r reproduces the natural topology order; the matmul must consume in the
                    matching order, else it deadlocks.
                global_cb (GlobalCircularBuffer): a DRAM-sender GCB (created via
                    ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher).
                    Supply exactly one of global_cb / prefetcher_pipes.
                prefetcher_pipes (List[PrefetcherPipe]): DRAM-sender PrefetcherPipes (created via
                    ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher) to deliver into
                    instead of a GCB. Receiver-contiguous tensors only; rotation works as it does
                    for a GCB. A tensor's per-receiver block size need not equal the pipes'
                    entry_size nor divide the ring, so size the ring for the consumer: one block is
                    enough for the transport, two for a consumer that keeps a block of lookahead.
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
        nb::arg("global_cb") = std::nullopt,
        nb::kw_only(),
        nb::arg("prefetcher_pipes") = std::vector<std::shared_ptr<tt::tt_metal::experimental::PrefetcherPipe>>{},
        nb::arg("device_subset") = std::nullopt,
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
        nb::arg("cq_id") = std::nullopt,
        nb::kw_only(),
        nb::arg("device_subset") = std::nullopt);

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

    // DRAM-sender GCB factories. MeshDevice-only (the per-mesh DRISC L1 arena lives on
    // MeshDeviceImpl) and only ever paired with the Tensor prefetcher above.
    ttnn::bind_function<"create_global_circular_buffer_for_tensor_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Create a GlobalCircularBuffer where senders are programmable DRAM cores (Blackhole DRISCs).
            Each bank id is mapped to an unused DRAM subchannel; receiver sets across senders must
            be disjoint and must not collide with the DRAM sender physical NOC coords.

            Args:
                mesh_device: The mesh device to create the buffer on.
                bank_to_receivers: List of (bank_id, receivers) pairs.
                size: Per-receiver fifo size in bytes.
                buffer_type: Buffer type (L1 or L1_SMALL).
                support_multi_receiver_shards: If True (default), a bank's shard may feed multiple
                    receivers (legacy interleaved layout), which requires a single sender per bank.
                    Set False to promise each receiver owns a disjoint contiguous shard
                    (receiver-contiguous layout); a bank with two or more receivers may then split
                    them across two DRISC sender cores for higher bandwidth.
        )doc",
        &ttnn::global_circular_buffer::create_global_circular_buffer_for_tensor_prefetcher,
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        nb::arg("support_multi_receiver_shards") = true);

    ttnn::bind_function<"create_prefetcher_pipes_for_tensor_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Create the PrefetcherPipes whose senders are programmable DRAM cores (Blackhole DRISCs),
            as an alternative Tensor prefetcher delivery target to a DRAM-sender
            GlobalCircularBuffer. Sender placement, the dual-sender receiver split, and slab
            numbering match create_global_circular_buffer_for_tensor_prefetcher, so a tensor laid
            out for one transport is laid out for the other.

            Returns one PrefetcherPipe per DRAM sender core, bank-major: a bank's pipes are
            adjacent, and the leading one owns that bank's leading receivers. That order is what
            assigns each sender its bank-local slab base, so pass the list on as it came.

            Consumers Attach the pipes and read them through the device-side PrefetcherPipe
            (wait_front / scoped_read_lock / pop_front). Keep the pipes alive for as long as any
            program uses them: dropping the last reference to one frees its ring and config.

            Args:
                mesh_device: The mesh device to create the buffer on.
                bank_to_receivers: List of (bank_id, receivers) pairs.
                entry_size: Push granularity in bytes the pipes start life at. With num_entries
                    it fixes the ring size, which never changes. A later queued tensor may use a
                    different per-receiver block size as long as the ring holds two of them.
                num_entries: Ring depth, in entries, per receiver.
                buffer_type: Buffer type (L1 or L1_SMALL).
                support_multi_receiver_shards: If True, a bank's shard may feed multiple receivers,
                    which forces a single sender per bank. Defaults to False (receiver-contiguous),
                    letting a bank with two or more receivers split them across two DRISC senders.
        )doc",
        &create_prefetcher_pipes_for_tensor_prefetcher_py,
        nb::arg("mesh_device"),
        nb::arg("bank_to_receivers"),
        nb::arg("entry_size"),
        nb::arg("num_entries"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        nb::arg("support_multi_receiver_shards") = false);

    ttnn::bind_function<"create_global_circular_buffer_for_matmul_1d", "ttnn.experimental.">(
        mod,
        R"doc(
            Build a DRAM-sender GlobalCircularBuffer sized to feed one or more gather_in0 or
            mcast_in0 1D matmuls with their weight tensors. The weight's DRAM layout is
            auto-detected (legacy WIDTH_SHARDED K-row-major vs receiver-contiguous NdShardSpec)
            and validated/sized accordingly.

            Args:
                mesh_device: The mesh device.
                program_configs: List of compatible 1D matmul program configs.
                weights: List of DRAM in1 tensors, one per program_config. All must share the same
                    DRAM layout (all legacy WIDTH_SHARDED, or all receiver-contiguous NdShardSpec).
                bank_to_receivers: List of (bank_id, receivers) pairs.
                size: GCB size in bytes.
                buffer_type: Buffer type (L1 or L1_SMALL).
                support_multi_receiver_shards: Optional per-bank sender-count override; leave None
                    (default) in production. When None the sender count follows the detected layout:
                    legacy WIDTH_SHARDED -> single sender per bank; receiver-contiguous -> dual
                    senders per bank (higher bandwidth; single-receiver banks fall back to one).
                    Pass an explicit value to override, mainly for tests/benchmarks: True forces
                    single sender, False forces dual senders. Forcing False (dual) on a legacy
                    weight raises (that layout is always single-sender).
        )doc",
        &ttnn::global_circular_buffer::create_global_circular_buffer_for_matmul_1d,
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("program_configs"),
        nb::arg("weights"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        nb::arg("support_multi_receiver_shards") = std::nullopt);

    ttnn::bind_function<"tensor_prefetcher_block_count_for_matmul_1d", "ttnn.experimental.">(
        mod,
        R"doc(
            Compute and validate the block_count to pair with a DRAM weight in
            queue_tensor_prefetcher_request for a gather_in0 or mcast_in0 1D matmul fed via
            global_cb. Gather returns the receiver/ring count. Mcast returns
            weight_K_tiles / in0_block_w and uses natural FIFO order.

            Args:
                program_config: The 1D matmul program config that will consume the weight.
                weight: The DRAM weight tensor, in either layout.
                global_cb: The DRAM-sender GCB the prefetcher and matmul share.

            Returns:
                int: the validated block_count.
        )doc",
        &ttnn::global_circular_buffer::tensor_prefetcher_block_count_for_matmul_1d,
        nb::arg("program_config"),
        nb::arg("weight"),
        nb::arg("global_cb"));
}

}  // namespace ttnn::operations::experimental
