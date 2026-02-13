// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "matmul_wo.hpp"

namespace ttnn::operations::experimental::deepseek::mla::detail {

void bind_matmul_wo(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::matmul_wo,
        R"doc(
        Experimental, high-performance Matmul WO operation for DeepSeek.

        Args:
            input_tensor: Input tensor (sharded)
            w_tensor: Weight tensor for matmul
            output_tensor: Output tensor (sharded)
            layer_id: The layer for which the Matmul WO operation is being performed
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("w_tensor"),
            nb::arg("output_tensor"),
            nb::arg("layer_id"),
        });
}

}  // namespace ttnn::operations::experimental::deepseek::mla::detail
