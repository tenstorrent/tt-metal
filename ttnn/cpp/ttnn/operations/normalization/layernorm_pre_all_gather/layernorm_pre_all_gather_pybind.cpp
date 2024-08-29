// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "layernorm_pre_all_gather.hpp"

namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_layernorm_pre_all_gather(py::module& module) {

    ttnn::bind_registered_operation(
        module,
        ttnn::layernorm_pre_all_gather,
        R"doc(layer_norm_pre_all_gather(input_tensor: ttnn.Tensor, program_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Compute layer_norm over :attr:`input_tensor`.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("program_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt}
    );
}

}  // namespace ttnn::operations::normalization::detail
