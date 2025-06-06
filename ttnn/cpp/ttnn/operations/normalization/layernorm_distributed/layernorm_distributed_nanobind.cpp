// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_distributed_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "layernorm_pre_all_gather.hpp"
#include "layernorm_post_all_gather.hpp"

namespace nb = nanobind;

namespace ttnn::operations::normalization::detail {

void bind_normalization_layernorm_pre_all_gather_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::layer_norm_pre_all_gather,
        R"doc(layer_norm_pre_all_gather(input_tensor: ttnn.Tensor, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor
            Compute sum(:attr:`input_tensor`ˆ2) and sum(:attr:`input_tensor`) over the last dimension.
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("residual_input_tensor") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
            nb::arg("program_config") = std::nullopt,
            nb::arg("memory_config") = std::nullopt});
}

void bind_normalization_layernorm_post_all_gather_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::layer_norm_post_all_gather,
        R"doc(layer_norm_post_all_gather(input_tensor: ttnn.Tensor, stats: ttnn.Tensor, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Performs the second part of a distributed layernorm operation normalizing the input based on the gathered statistics input.
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("stats"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = std::nullopt,
            nb::arg("bias") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
            nb::arg("program_config") = std::nullopt,
            nb::arg("dtype") = std::nullopt});
}

void bind_normalization_layernorm_distributed(nb::module_& mod) {
    bind_normalization_layernorm_pre_all_gather_operation(mod);
    bind_normalization_layernorm_post_all_gather_operation(mod);
}

}  // namespace ttnn::operations::normalization::detail
