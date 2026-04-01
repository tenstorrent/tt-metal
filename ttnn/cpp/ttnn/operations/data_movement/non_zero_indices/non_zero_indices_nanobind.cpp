// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "non_zero_indices_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "non_zero_indices.hpp"

namespace ttnn::operations::data_movement {
namespace {

void bind_non_zero(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns the number of elements (N) that are non-zero as well as a tensor of the same shape as input where the first N elements are the indices of non-zero elements.

        Args:
            input_tensor (ttnn.Tensor): Input Tensor should be 1D and in row major layout.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensors.
    )doc";

    ttnn::bind_function<"nonzero">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::nonzero, nb::arg("input_tensor").noconvert(), nb::kw_only(), nb::arg("memory_config") = nb::none()));
}

}  // namespace

void bind_non_zero_indices(nb::module_& mod) { bind_non_zero(mod); }

}  // namespace ttnn::operations::data_movement
