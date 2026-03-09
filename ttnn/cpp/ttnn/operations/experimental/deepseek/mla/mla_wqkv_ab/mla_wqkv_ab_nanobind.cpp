// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_wqkv_ab_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "mla_wqkv_ab.hpp"

namespace ttnn::operations::experimental::deepseek::mla::mla_wqkv_ab::detail {

void bind_mla_wqkv_ab(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::deepseek::mla::mla_wqkv_ab,
        R"doc(
        Experimental, high-performance MLA WqkvAb operation for DeepSeek.

        Args:
            input_tensor: Input tensor (sharded)
            w_tensor: Weight tensor
            output_tensor: Output tensor (sharded)
            layer_id: The layer for which the MLA WqkvAb operation is being performed
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("w_tensor"),
            nb::arg("output_tensor"),
            nb::arg("layer_id"),
        });
}

}  // namespace ttnn::operations::experimental::deepseek::mla::mla_wqkv_ab::detail

namespace ttnn::operations::experimental::deepseek::mla::detail {

void bind_mla_wqkv_ab(::nanobind::module_& mod) { mla_wqkv_ab::detail::bind_mla_wqkv_ab(mod); }

}  // namespace ttnn::operations::experimental::deepseek::mla::detail
