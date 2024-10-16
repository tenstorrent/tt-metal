// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_distributed_pybind.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "layernorm_pre_all_gather.hpp"
#include "layernorm_post_all_gather.hpp"

namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_layernorm_pre_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::layer_norm_pre_all_gather,
        R"doc(layer_norm_pre_all_gather(input_tensor: ttnn.Tensor, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor
            Compute sum(:attr:`input_tensor`ˆ2) and sum(:attr:`input_tensor`) over the last dimension.
        )doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor"),
                                 py::kw_only(),
                                 py::arg("dtype") = DataType::BFLOAT16,
                                 py::arg("compute_kernel_config") = std::nullopt,
                                 py::arg("program_config") = std::nullopt,
                                 py::arg("memory_config") = std::nullopt});
}

void bind_normalization_layernorm_post_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::layer_norm_post_all_gather,
        R"doc(layer_norm_post_all_gather(input_tensor: ttnn.Tensor, stats: ttnn.Tensor, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Performs the second part of a distributed layernorm operation normalizing the input based on the gathered statistics input.
        )doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor"),
                                 py::arg("stats"),
                                 py::kw_only(),
                                 py::arg("epsilon") = 1e-12,
                                 py::arg("weight") = std::nullopt,
                                 py::arg("bias") = std::nullopt,
                                 py::arg("memory_config") = std::nullopt,
                                 py::arg("compute_kernel_config") = std::nullopt,
                                 py::arg("program_config") = std::nullopt});
}

void bind_normalization_layernorm_distributed(py::module& module) {
    bind_normalization_layernorm_pre_all_gather_operation(module);
    bind_normalization_layernorm_post_all_gather_operation(module);
}

}  // namespace ttnn::operations::normalization::detail
