// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumsum_pybind.hpp"
#include <optional>
#include "pybind11/decorators.hpp"
#include <pybind11/stl.h>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/device/cumsum_device_operation.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::reduction::detail {
namespace py = pybind11;

void bind_cumsum_operation(py::module& module) {
    auto docstring =
        R"doc(
        Returns cumulative sum of `input` along dimension `dim`

        For a given `input` of size N, the `output` will also contain N elements and be such that:

        ``y_i = x_0 + x_1 + ... = x_{i-1} + x_i``

        This function is fundamentally identical to `torch.cumsum()`

        Parameters:
            * `input` (ttnn.Tensor) input tensor
            * `dim` (int)

        Keywords Arguments:
            * `dtype` (ttnn.DataType, optional) desired output type. If specified then input tensor will be casted to `dtype` before processing.
            * `output` (ttnn.Tensor, optional) preallocated output. If specified, `output` must have same shape as `input`, and must be on the same device.

        Note:
            If both `dtype` and `output` are specified then `output.dtype` must be `dtype`

        Example:

        .. code-block:: python
            import torch
            import ttnn

            # Create tensor
            torch_input = torch.rand([2, 3, 4])
            tensor_input = ttnn.from_torch(torch_input, device=device)

            # Apply `ttnn.experimental.cumsum()` on `dim=0`
            tensor_output = ttnn.experimental.cumsum(tensor_input, dim=0)

            # With preallocated output and dtype
            preallocated_output = ttnn.from_torch(torch.rand([2, 3, 4]), dtype=ttnn.bfloat16, device=device)

            tensor_output = ttnn.experimental.cumsum(tensor_input, dim=0, dtype=torch.bfloat16, output=preallocated_output)
        )doc";

    using OperationType = decltype(ttnn::experimental::cumsum);
    bind_registered_operation(
        module,
        ttnn::experimental::cumsum,
        "ttnn.experimental.cumsum()",
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int64_t dim,
               std::optional<tt::tt_metal::DataType>& dtype,
               std::optional<Tensor> preallocated_tensor,
               QueueId queue_id) { return self(queue_id, input_tensor, dim, dtype, preallocated_tensor); },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("output") = std::nullopt,
            py::arg("queueId") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::reduction::detail
