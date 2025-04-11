// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"

#include "concat.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_concat(nb::module_& mod) {
    const auto doc = R"doc(

Args:
    input_tensor (List of ttnn.Tensor): the input tensors.
    dim (number): the concatenating dimension.

Keyword Args:
    memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
    output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
    groups (int, optional): When `groups` is set to a value greater than 1, the inputs are split into N `groups` partitions, and elements are interleaved from each group into the output tensor. Each group is processed independently, and elements from each group are concatenated in an alternating pattern based on the number of groups. This is useful for recombining grouped convolution outputs during residual concatenation. Defaults to `1`. Currently, groups > `1` is only supported for two height sharded input tensors.

Returns:
    ttnn.Tensor: the output tensor.

Example:

    >>> tensor1 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> tensor2 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> output = ttnn.concat([tensor1, tensor2], dim=3)
    >>> print(output.shape)
    [1, 1, 64, 64]

    )doc";

    using OperationType = decltype(ttnn::concat);
    ttnn::bind_registered_operation(
        mod,
        ttnn::concat,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const std::vector<ttnn::Tensor>& tensors,
               const int dim,
               std::optional<ttnn::Tensor>& optional_output_tensor,
               std::optional<ttnn::MemoryConfig>& memory_config,
               const int groups) { return self(tensors, dim, memory_config, optional_output_tensor, groups); },
            nb::arg("tensors"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("output_tensor").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("groups") = 1});
}

}  // namespace ttnn::operations::data_movement::detail
