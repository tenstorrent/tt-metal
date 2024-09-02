// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "layernorm_post_all_gather.hpp"

namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_layernorm_post_all_gather(py::module& module) {

    ttnn::bind_registered_operation(
        module,
        ttnn::layernorm_post_all_gather,
        R"doc(layernorm_post_all_gather(input_tensor: ttnn.Tensor, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None, residual_input_tensor: Optional[ttnn.Tensor] = None, E_x: Optional[ttnn.Tensor] = None, E_x2: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None, program_config: Optional[ttnn.ProgramConfig] = None) -> ttnn.Tensor
            Compute layer_norm over :attr:`input_tensor` based on pre-computed statistrics E(x) and E(x^2). Expects E(x) and E(x^2) of multiple devices interleaved in the width dimention, where each E(x) and E(xˆ2) is of shape [1, 32] containing the respective value in the first column.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("epsilon") = 1e-12,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("residual_input_tensor") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("stats") = std::nullopt});
}

}  // namespace ttnn::operations::normalization::detail
