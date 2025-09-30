// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "repeat.hpp"

namespace ttnn::operations::data_movement {
namespace nb = nanobind;

namespace detail {
template <typename data_movement_operation_t>
void bind_repeat(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    ttnn::bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<uint32_t>& repetition_vector,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(input_tensor, repetition_vector, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("repeat_dims"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace detail

void bind_repeat(nb::module_& mod) {
    auto doc = R"doc(

    Returns a new tensor filled with repetition of input :attr:`input_tensor` according to number of times specified in :attr:`shape`.

    Args:
        input_tensor (ttnn.Tensor): the input tensor.
        repetition_vector (SmallVector): The number of repetitions for each dimension.

    Keyword Args:
        memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

    Returns:
        ttnn.Tensor: the output tensor.

    Example:

        >>> tensor = ttnn.repeat(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]]), [1,2],)), device)
        >>> print(tensor)
        tensor([[1, 2],
        [1, 2],
        [3, 4],
        [3, 4]])
            )doc";

    detail::bind_repeat(mod, ttnn::repeat, doc);
}

}  // namespace ttnn::operations::data_movement
