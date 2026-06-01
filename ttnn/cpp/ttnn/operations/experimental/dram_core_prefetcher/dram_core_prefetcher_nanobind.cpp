// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_core_prefetcher_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "dram_core_prefetcher.hpp"
#include "ttnn/global_circular_buffer.hpp"

namespace ttnn::operations::experimental {

void bind_dram_core_prefetcher(nb::module_& mod) {
    ttnn::bind_function<"start_dram_core_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Start the queueable DRAM-core (DRISC) prefetcher on `mesh_device`. Returns
            immediately; one DRISC kernel parks on a per-(device, sender-core) H2D socket
            waiting for requests. Pair with queue_dram_core_prefetcher_request and
            stop_dram_core_prefetcher.

            Only one DRAM-core prefetcher may be active per mesh device at a time.
            Receiver count is per-GCB (read from each GCB's sender state block on every
            request), so a single prefetcher can serve GCBs with different num_receivers
            values.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device to launch on.
        )doc",
        &start_dram_core_prefetcher,
        nb::arg("mesh_device"));

    ttnn::bind_function<"queue_dram_core_prefetcher_request", "ttnn.experimental.">(
        mod,
        R"doc(
            Queue one prefetch request. Non-blocking. Per-GCB ring-buffer state is
            preserved across requests, so successive Queue calls against the same GCB
            resume where the previous call left off. Successive Queue calls can target
            different GCBs.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device whose prefetcher to queue on.
                tensors (List[Tuple[ttnn.Tensor, int]]): the full, flattened list of
                    (weight tensor, block_count) pairs to prefetch (at least one), streamed
                    in list order. block_count is the number of K-blocks to divide that
                    tensor's K dimension into (the consumer matmul waits on block_count
                    pages per layer). Pass distinct tensors for distinct layers, or repeat
                    a tensor to replay it.
                global_cb (GlobalCircularBuffer): a DRAM-sender GCB (created via
                    ttnn.experimental.create_global_circular_buffer_with_dram_senders).
                device_subset (Optional[MeshCoordinateRangeSet]): subset of the mesh that
                    processes this request. Defaults to the full mesh.

            Returns:
                None
        )doc",
        &queue_dram_core_prefetcher_request,
        nb::arg("mesh_device"),
        nb::arg("tensors"),
        nb::arg("global_cb"),
        nb::kw_only(),
        nb::arg("device_subset") = std::nullopt);

    ttnn::bind_function<"stop_dram_core_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Push the stop sentinel to every socket, join the host worker thread, and wait
            for the kernels to exit. No-op if no prefetcher is active.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device whose prefetcher to stop.
        )doc",
        &stop_dram_core_prefetcher,
        nb::arg("mesh_device"));

    // DRAM-sender GCB factories. MeshDevice-only (the per-mesh DRISC L1 arena lives on
    // MeshDeviceImpl) and only ever paired with the DRAM-core prefetcher above.
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
        )doc",
        &ttnn::global_circular_buffer::create_global_circular_buffer_with_dram_senders,
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1);

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
}

}  // namespace ttnn::operations::experimental
