// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_prefetcher_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "tensor_prefetcher.hpp"
#include "ttnn/global_circular_buffer.hpp"

namespace ttnn::operations::experimental {

void bind_tensor_prefetcher(nb::module_& mod) {
    ttnn::bind_function<"is_tensor_prefetcher_supported", "ttnn.experimental.">(
        mod,
        R"doc(
            Return True if the Tensor prefetcher (DRISC) is supported on `mesh_device`,
            i.e. programmable DRAM cores are available (Blackhole with firmware >= 19.12.0.0
            and either no harvested DRAM channels or a single device). When this returns False,
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
                    ttnn.experimental.create_global_circular_buffer_with_dram_senders).
                device_subset (Optional[MeshCoordinateRangeSet]): subset of the mesh that
                    processes this request. Defaults to the full mesh.
                cq_id (Optional[int]): command queue that may be recording a trace. When that
                    CQ is mid trace-capture, the request is captured into the trace instead of
                    being sent immediately, and is re-sent on every execute_trace of that trace.
                    Defaults to the current/default command queue.

            Returns:
                None
        )doc",
        &queue_tensor_prefetcher_request,
        nb::arg("mesh_device"),
        nb::arg("tensors"),
        nb::arg("global_cb"),
        nb::kw_only(),
        nb::arg("device_subset") = std::nullopt,
        nb::arg("cq_id") = std::nullopt);

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
                cq_id (int): the command queue to fence against.
                device_subset (Optional[MeshCoordinateRangeSet]): subset of the mesh to
                    fence. Defaults to the full mesh.

            Returns:
                None
        )doc",
        &wait_for_cq_on_tensor_prefetcher,
        nb::arg("mesh_device"),
        nb::arg("cq_id") = 0,
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
    ttnn::bind_function<"create_global_circular_buffer_with_dram_senders", "ttnn.experimental.">(
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
        &ttnn::global_circular_buffer::create_global_circular_buffer_with_dram_senders,
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        nb::arg("support_multi_receiver_shards") = true);

    ttnn::bind_function<"create_global_circular_buffer_for_matmul_1d", "ttnn.experimental.">(
        mod,
        R"doc(
            Build a DRAM-sender GlobalCircularBuffer sized to feed one or more 1D ring matmuls
            (gather_in0=true) with their weight tensors. See impl notes in
            tt_metal/impl/buffers/global_circular_buffer.cpp.

            Args:
                mesh_device: The mesh device.
                program_configs: List of 1D mcast matmul program configs (each gather_in0=True).
                weights: List of DRAM-sharded in1 tensors, one per program_config.
                bank_to_receivers: List of (bank_id, receivers) pairs.
                size: GCB size in bytes.
                buffer_type: Buffer type (L1 or L1_SMALL).
        )doc",
        &ttnn::global_circular_buffer::create_global_circular_buffer_for_matmul_1d,
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("program_configs"),
        nb::arg("weights"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1);

    ttnn::bind_function<"tensor_prefetcher_block_count_for_matmul_1d", "ttnn.experimental.">(
        mod,
        R"doc(
            Compute and validate the block_count to pair with a receiver-contiguous DRAM weight
            in queue_tensor_prefetcher_request, for a gather_in0 1D matmul fed via global_cb.
            Centralizes the recv-contig prefetcher/matmul cross-checks (num_shards == ring_size,
            weight K divisible by ring_size, weight per-receiver N == per_core_N) so call sites
            don't re-derive (and mis-derive) them. Returns the block_count (== ring_size); raises
            on any mismatch.

            Args:
                program_config: The gather_in0 1D mcast matmul program config that will consume the weight.
                weight: The receiver-contiguous (NdShardSpec) DRAM weight tensor.
                global_cb: The DRAM-sender GCB the prefetcher and matmul share.

            Returns:
                int: the validated block_count.
        )doc",
        &ttnn::global_circular_buffer::tensor_prefetcher_block_count_for_matmul_1d,
        nb::arg("program_config"),
        nb::arg("weight"),
        nb::arg("global_cb"));

    ttnn::bind_function<"create_global_circular_buffer_for_matmul_1d_recv_contig", "ttnn.experimental.">(
        mod,
        R"doc(
            Receiver-contiguous counterpart of create_global_circular_buffer_for_matmul_1d: build a
            DRAM-sender GlobalCircularBuffer sized to feed one or more gather_in0 1D ring matmuls from
            NdShardSpec (receiver-contiguous) DRAM weights, validating the (program_config, weight,
            bank_to_receivers) triple in one place. See impl notes in ttnn/core/global_circular_buffer.cpp.

            Args:
                mesh_device: The mesh device.
                program_configs: List of 1D mcast matmul program configs (each gather_in0=True).
                weights: List of NdShardSpec DRAM in1 tensors, one per program_config.
                bank_to_receivers: List of (bank_id, receivers) pairs (strided round-robin placement).
                size: GCB size in bytes (>= ring_size * largest per-receiver page).
                buffer_type: Buffer type (L1 or L1_SMALL).
                support_multi_receiver_shards: If True (default), a bank's shard may feed multiple
                    receivers (single sender per bank). Set False to promise each receiver owns a
                    disjoint contiguous shard, letting a bank split its receivers across two DRISC
                    sender cores for higher bandwidth.
        )doc",
        &ttnn::global_circular_buffer::create_global_circular_buffer_for_matmul_1d_recv_contig,
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("program_configs"),
        nb::arg("weights"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        nb::arg("support_multi_receiver_shards") = true);
}

}  // namespace ttnn::operations::experimental
