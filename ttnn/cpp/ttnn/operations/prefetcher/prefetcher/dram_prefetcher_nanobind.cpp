// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "dram_prefetcher.hpp"
#include "dram_core_prefetcher.hpp"

namespace ttnn::operations::dram_prefetcher::detail {

void bind_dram_prefetcher_operation(nb::module_& mod) {
    ttnn::bind_function<"dram_prefetcher">(
        mod,
        R"doc(
            Asynchronously pre-fetch tensors from DRAM into the neighbouring L1 cores.
            This utilizes a global circular buffer to push data on consumer cores.

            Args:
                tensors (List[ttnn.Tensor]): A list of tensor objects to be pre-fetched.
                tensor_addrs (ttnn.Tensor): A tensor (row major layout) that contains memory addresses corresponding to the tensor locations in DRAM. The format should be as follows: [t1_l1, t2_l1, ..., t1_l2, t2_l2, ..., t1_l3, t2_l3, ...]
                num_layers (int): The number of layers in the pipeline or the model for which tensors need to be pre-fetched.
                global_cb (GlobalCircularBuffer): A global cb object, used internally to manage data movement across dram reader cores, and downstream consumer cores.
                enable_performance_mode (bool, optional): If set to true, the operation will be optimized for performance. May lead to ND behavior on wormhole 4U systems!

            Returns:
                ttnn.Tensor: empty tensor (TODO: Should return None)
        )doc",
        &ttnn::dram_prefetcher,
        nb::arg("tensors"),
        nb::arg("num_layers"),
        nb::arg("global_cb") = std::nullopt,
        nb::kw_only(),
        nb::arg("enable_performance_mode") = false,
        nb::arg("dram_core_k_block_w_tiles") = 1);
}

void bind_dram_core_prefetcher_lifecycle(nb::module_& mod) {
    mod.def(
        "start_dram_core_prefetcher",
        &ttnn::start_dram_core_prefetcher,
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
                    (created via ttnn.create_global_circular_buffer_with_dram_senders).
                enable_performance_mode (bool, optional): kept for API parity; currently a no-op.
                dram_core_k_block_w_tiles (int, optional): K-block width in tiles. Default 1.

            Returns:
                None
        )doc",
        nb::arg("mesh_device"),
        nb::arg("tensors"),
        nb::arg("num_layers"),
        nb::arg("global_cb"),
        nb::kw_only(),
        nb::arg("enable_performance_mode") = false,
        nb::arg("dram_core_k_block_w_tiles") = 1);

    mod.def(
        "stop_dram_core_prefetcher",
        &ttnn::stop_dram_core_prefetcher,
        R"doc(
            Block until the active DRAM-core prefetcher finishes its num_layers loop, then
            release its resources. No-op if no prefetcher is active.

            Callers invoke stop after enqueuing all consuming matmul programs; stop is what
            drains the pipeline.

            Args:
                mesh_device (ttnn.MeshDevice): the mesh device whose prefetcher to stop.
        )doc",
        nb::arg("mesh_device"));
}

void bind_dram_prefetcher(nb::module_& mod) {
    bind_dram_prefetcher_operation(mod);
    bind_dram_core_prefetcher_lifecycle(mod);
}

}  // namespace ttnn::operations::dram_prefetcher::detail
