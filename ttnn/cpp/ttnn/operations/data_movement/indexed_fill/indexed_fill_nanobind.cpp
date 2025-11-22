// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_fill_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "indexed_fill.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

void bind_indexed_fill(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
            Replaces batch of input in input_b denoted by batch_ids into input_a.

            Args:
                batch_id (ttnn.Tensor): the input tensor.
                input_tensor_a (ttnn.Tensor): the input tensor.
                input_tensor_b (ttnn.Tensor): the input tensor.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                dim (int, optional): Dimension value. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.

            Example:
                >>> batch_id = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.UINT32)), device=device)
                >>> input_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
                >>> input_b = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
                >>> output = ttnn.indexed_fill(batch_id, tensor1, tensor2)
        )doc",
        ttnn::indexed_fill.base_name());

    using OperationType = decltype(ttnn::indexed_fill);
    ttnn::bind_registered_operation(
        mod,
        ttnn::indexed_fill,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& batch_id,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               int64_t dim) { return self(batch_id, input_tensor_a, input_tensor_b, memory_config, dim); },
            nb::arg("batch_id").noconvert(),
            nb::arg("input_tensor_a").noconvert(),
            nb::arg("input_tensor_b").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dim") = 0});
}

}  // namespace detail

}  // namespace ttnn::operations::data_movement
