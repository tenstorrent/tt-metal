// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cumsum_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::reduction::accumulation::detail {
namespace py = pybind11;

void bind_reduction_cumsum_operation(py::module& module) {
    auto docstring =
        R"doc(
        Returns cumulative sum of `input` along dimension `dim`
        For a given `input` of size N, the `output` will also contain N elements and be such that:
        This function is fundamentally identical to `torch.cumsum()`

        Args:
            input (ttnn.Tensor): input tensor
            dim (int): dimension along which to compute cumulative sum

        Keyword Args:
            dtype (ttnn.DataType, optional): desired output type. If specified then input tensor will be casted to `dtype` before processing.
            reverse_order (bool, optional, default False): whether to perform accumulation from the end to the beginning of accumulation axis.
            output (ttnn.Tensor, optional): preallocated output. If specified, `output` must have same shape as `input`, and must be on the same device.

        Returns:
            ttnn.Tensor: the output tensor.



        Note:
            If both `dtype` and `output` are specified then `output.dtype` must be `dtype`)

            Supported dtypes, layout, ranks and `dim` values:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
                 - dim
               * - BFLOAT16, FLOAT32
                 - TILE
                 - 1, 2, 3, 4, 5
                 - -rank <= dim < rank
               * - INT32, UINT32
                 - TILE
                 - 3, 4, 5
                 - dim in {0, 1, ..., rank - 3} or dim in {-rank, -rank + 1, ..., -3}

        Example:

        .. code-block:: python

            import torch
            import ttnn

            # Create tensor
            torch_input = torch.rand([2, 3, 4])
            tensor_input = ttnn.from_torch(torch_input, device=device)

            # Apply ttnn.cumsum() on dim=0
            tensor_output = ttnn.cumsum(tensor_input, dim=0)

            # With preallocated output and dtype
            preallocated_output = ttnn.from_torch(torch.rand([2, 3, 4]), dtype=ttnn.bfloat16, device=device)

            tensor_output = ttnn.cumsum(tensor_input, dim=0, dtype=torch.bfloat16, out=preallocated_output)
        )doc";

    using OperationType = decltype(ttnn::cumsum);
    bind_registered_operation(
        module,
        ttnn::cumsum,
        docstring,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t& dim,
               std::optional<DataType>& dtype,
               const bool& reverse_order,
               std::optional<Tensor> preallocated_tensor,
               QueueId queue_id) {
                return self(queue_id, input_tensor, dim, dtype, reverse_order, preallocated_tensor);
            },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("reverse_order") = false,
            py::arg("out") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::reduction::accumulation::detail
