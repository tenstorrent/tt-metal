// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "concat.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_concat(nb::module_& mod) {
    const std::string doc = R"doc(

        Args:
            input_tensor (List of ttnn.Tensor): the input tensors.
            dim (number): the concatenating dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            groups (int, optional): When `groups` is set to a value greater than 1, the inputs are split into N `groups` partitions, and elements are interleaved from each group into the output tensor. Each group is processed independently, and elements from each group are concatenated in an alternating pattern based on the number of groups. This is useful for recombining grouped convolution outputs during residual concatenation. Defaults to `1`. Currently, groups > `1` is only supported for two height sharded input tensors.

        Keyword Args:
            sub_core_grids (ttnn.CoreRangeSet, optional): Sub-core grid to use for interleaved (L1 or DRAM) output tensors. If provided, the concatenation will run on the specified sub-core grid instead of the full compute grid. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"concat">(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            &ttnn::concat,
            nb::arg("tensors"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor").noconvert() = nb::none(),
            nb::arg("groups") = 1,
            nb::arg("sub_core_grids") = nb::none()));
}

}  // namespace ttnn::operations::data_movement::detail
