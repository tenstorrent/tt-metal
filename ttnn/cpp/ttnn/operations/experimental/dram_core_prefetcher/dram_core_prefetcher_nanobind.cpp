// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_core_prefetcher_nanobind.hpp"

#include <nanobind/nanobind.h>
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
            Start the DRAM-core (DRISC) prefetcher on `mesh_device`. Returns immediately;
            the kernel runs asynchronously on its DRISC core(s) for `num_layers` iterations,
            pushing each tensor in `tensors` into the receivers configured in `global_cb`.

            Only one DRAM-core prefetcher may be active per mesh device at a time; calling
            start again before stop will raise.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device to launch on.
                tensors (List[ttnn.Tensor]): data tensors followed by a trailing tensor_addrs
                    tensor. The addrs tensor is unused on the DRAM-core path but kept for
                    shape parity with ttnn.dram_prefetcher.
                num_layers (int): number of prefetch iterations the kernel will run.
                global_cb (GlobalCircularBuffer): must be a DRAM-sender GCB
                    (created via ttnn.experimental.create_global_circular_buffer_with_dram_senders).

            Returns:
                None
        )doc",
        &start_dram_core_prefetcher,
        nb::arg("mesh_device"),
        nb::arg("tensors"),
        nb::arg("num_layers"),
        nb::arg("global_cb"));

    ttnn::bind_function<"stop_dram_core_prefetcher", "ttnn.experimental.">(
        mod,
        R"doc(
            Block until the active DRAM-core prefetcher finishes its num_layers loop, then
            release its resources. No-op if no prefetcher is active.

            Callers invoke stop after enqueuing all consuming matmul programs; stop is what
            drains the pipeline.

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
            (gather_in0=true) with their weight tensors. Size, page stride, and the
            receiver-to-bank rectangle are derived from the matmul program configs so
            host-side alignment checks fire as TT_FATAL at construction rather than as a
            silent device hang during ttnn.linear.

            One (program_config, weight) pair per matmul. The production pattern (e.g. llama 70B)
            is to share a single GCB across XQKV/WO/FF1/FF2, where each consumer has a different
            in1_block_size.

            Validates per matmul:
              * weight K is tile-aligned AND divisible by ring_size (no activation padding past K),
              * weight N shards evenly across DRAM banks and per-bank N splits evenly across
                num_global_cb_receivers,
              * matmul per_core_N == weight per-receiver N.
            All configs must agree on compute_with_storage_grid_size and num_global_cb_receivers.

            Picking size (in bytes):
              * Minimum = num_blocks * largest_in1_block_size (one full layer's pages).
                The matmul does wait_front(num_blocks) per layer, so anything smaller deadlocks.
              * Larger lets the DRISC prefetcher run further ahead. Past one full layer
                worth more buffering doesn't add throughput; the DRISC just stalls on
                remote_cb_reserve_back.
              * The factory does not check L1 capacity. Callers must size the GCB to fit
                alongside the matmul's in0/in1/out/interm CBs on the receiver cores.

            Args:
                mesh_device: The mesh device.
                program_configs: List of 1D mcast matmul program configs (each gather_in0=True).
                weights: List of DRAM-sharded in1 tensors, one per program_config.
                bank_to_receivers: List of (bank_id, receivers) pairs. num_senders *
                    num_global_cb_receivers must equal ring_size; every bank must own
                    exactly num_global_cb_receivers receiver cores. The matmul op asserts
                    at construction time that the receivers row-major-walk into the same
                    order as its activation grid.
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
