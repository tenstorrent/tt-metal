// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "non_zero_indices_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "non_zero_indices.hpp"

namespace nb = nanobind;

namespace ttnn::operations::data_movement {

void bind_non_zero_indices(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns the number of elements (N) that are non-zero as well as a tensor of the same shape as input where the first N elements are the indices of non-zero elements.

        Args:
        input_tensor (ttnn.Tensor): Input Tensor should be 1D and in row major layout.

        Keyword Args:
        memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
        Tuple of ttnn.Tensor: the output tensors (count, indices).
    )doc";

    mod.def(
        "nonzero",
        &ttnn::nonzero,
        doc,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::data_movement
