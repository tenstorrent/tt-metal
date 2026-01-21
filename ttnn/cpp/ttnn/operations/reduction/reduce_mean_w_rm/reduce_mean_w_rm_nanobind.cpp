// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_mean_w_rm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "reduce_mean_w_rm.hpp"

namespace ttnn::operations::reduce_mean_w_rm {

void bind_reduce_mean_w_rm_operation(nb::module_& mod) {
    const auto doc =
        R"doc(Computes the arithmetic mean across the width (last) dimension of a row-major tensor. The output tensor has the same shape as input except the last dimension becomes 1 (logically), padded to 32 (physically).)doc";

    bind_registered_operation(
        mod,
        ttnn::reduce_mean_w_rm,
        doc,
        ttnn::nanobind_arguments_t{nb::arg("input_tensor"), nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::reduce_mean_w_rm
