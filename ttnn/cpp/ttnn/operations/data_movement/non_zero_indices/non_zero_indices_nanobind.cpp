// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "non_zero_indices_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "non_zero_indices.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

void bind_non_zero(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(

            Returns the number of elements (N) that are non-zero as well as a tensor of the same shape as input where the first N elements are the indices of non-zero elements.

            Args:
                input_tensor (ttnn.Tensor): Input Tensor should be 1D and in row major layout.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                List of ttnn.Tensor: the output tensors.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.nonzero(tensor)
        )doc",
        ttnn::nonzero.base_name());

    using OperationType = decltype(ttnn::nonzero);
    ttnn::bind_registered_operation(
        mod,
        ttnn::nonzero,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, input_tensor, memory_config); },
            nb::arg("input_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

}  // namespace detail

void bind_non_zero_indices(nb::module_& mod) { detail::bind_non_zero(mod); }

}  // namespace ttnn::operations::data_movement
