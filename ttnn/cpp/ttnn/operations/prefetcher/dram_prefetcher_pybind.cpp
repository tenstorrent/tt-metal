// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_pybind.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "dram_prefetcher.hpp"

namespace ttnn::operations::dram_prefetcher::detail {

namespace py = pybind11;

void bind_dram_prefetcher_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::dram_prefetcher,
        R"doc(
            Asyncroneously pre-fetch tensors from DRAM and signal completion through semaphores.

        Args:
            tensors (List[ttnn.Tensor]): the tensors to pre-fetch.

        )doc",

        ttnn::pybind_arguments_t{
            py::arg("tensors"),
            // py::kw_only(),
        });
}

void bind_dram_prefetcher(py::module& module) { bind_dram_prefetcher_operation(module); }

}  // namespace ttnn::operations::dram_prefetcher::detail
