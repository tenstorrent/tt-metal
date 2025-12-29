// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "moe.hpp"

namespace ttnn::operations::experimental::moe::detail {

void bind_moe(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::moe,
        R"doc(
        moe(input_tensor, weight_tensor)

        Experimental, high-performance MoE operation.
        )doc",
        ttnn::nanobind_arguments_t{nb::arg("input_tensor"), nb::arg("weight_tensor")});
}

}  // namespace ttnn::operations::experimental::moe::detail
