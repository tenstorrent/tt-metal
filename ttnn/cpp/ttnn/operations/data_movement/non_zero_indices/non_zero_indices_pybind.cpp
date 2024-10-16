// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "non_zero_indices_pybind.hpp"
#include "non_zero_indices.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::data_movement {
namespace detail {
namespace py = pybind11;

void bind_non_zero(py::module& module) {
    auto doc = fmt::format(
        R"doc(

            Returns the number of elements (N) that are non-zero as well as a tensor of the same shape as input where the first N elements are the indices of non-zero elements.

            Args:
                input_tensor (ttnn.Tensor): Input Tensor should be 1D and in row major layout.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                List of ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.nonzero(tensor)
        )doc",
        ttnn::nonzero.base_name());

    using OperationType = decltype(ttnn::nonzero);
    ttnn::bind_registered_operation(module,
                                    ttnn::nonzero,
                                    doc,
                                    ttnn::pybind_overload_t{[](const OperationType& self,
                                                               const ttnn::Tensor& input_tensor,
                                                               const std::optional<ttnn::MemoryConfig>& memory_config,
                                                               uint8_t queue_id) {
                                                                return self(queue_id, input_tensor, memory_config);
                                                            },
                                                            py::arg("input_tensor").noconvert(),
                                                            py::kw_only(),
                                                            py::arg("memory_config") = std::nullopt,
                                                            py::arg("queue_id") = 0});
}

}  // namespace detail

void bind_non_zero_indices(py::module& module) {
    detail::bind_non_zero(module);
}

}  // namespace ttnn::operations::data_movement
