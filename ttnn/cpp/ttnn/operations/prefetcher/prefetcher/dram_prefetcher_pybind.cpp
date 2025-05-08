// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_pybind.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "dram_prefetcher.hpp"

namespace ttnn::operations::dram_prefetcher::detail {

namespace py = pybind11;

void bind_dram_prefetcher_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::dram_prefetcher,
        R"doc(
            Asynchronously pre-fetch tensors from DRAM into the neighbouring L1 cores.
            This utilizes a global circular buffer to push data on consumer cores.

            Args:
                tensors (List[ttnn.Tensor]): A list of tensor objects to be pre-fetched.
                tensor_addrs (ttnn.Tensor): A tensor (row major layout) that contains memory addresses
                    corresponding to the tensor locations in DRAM. The format should be as follows:
                        [t1_l1, t2_l1, ..., t1_l2, t2_l2, ..., t1_l3, t2_l3, ...]
                num_layers (int): The number of layers in the pipeline or the model
                    for which tensors need to be pre-fetched.
                global_cb (GlobalCircularBuffer): A global cb object, used internally to manage data movement
                    across dram reader cores, and downstream consumer cores.
                enable_performance_mode (bool, optional): If set to true, the operation will be optimized for performance.
                    May lead to ND behavior on wormhole 4U systems!

            Returns:
                ttnn.Tensor: empty tensor (TODO: Should return None)
        )doc",

        ttnn::pybind_arguments_t{
            py::arg("tensors"),
            py::arg("num_layers"),
            py::arg("global_cb"),
            py::kw_only(),
            py::arg("enable_performance_mode") = false,
        });
}

void bind_dram_prefetcher(py::module& module) { bind_dram_prefetcher_operation(module); }

}  // namespace ttnn::operations::dram_prefetcher::detail
