// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/reduction/accumulation/cumprod/cumprod.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::reduction::accumulation::detail {
void bind_reduction_cumprod_operation(py::module& module) {
    auto docstring =
        R"doc(
        Returns cumulative product of `input` along dimension `dim`
        For a given `input` of size N, the `output` will also contain N elements and be such that:

        .. math::
            \mathrm{{output}}_i = \mathrm{{input}}_1 \times \mathrm{{input}}_2 \times \cdots \times \mathrm{{input}}_i


        Args:
            input (ttnn.Tensor): input tensor. Must be on the device.
            dim (int): dimension along which to compute cumulative product

        Keyword Args:
            dtype (ttnn.DataType, optional): desired output type. If specified then input tensor will be casted to `dtype` before processing.
            reverse_order (bool, optional, default False): whether to perform accumulation from the end to the beginning of accumulation axis.
            out (ttnn.Tensor, optional): preallocated output. If specified, `out` must have same shape as `input`, and must be on the same device.

        Note:
            If both `dtype` and `output` are specified then `output.dtype` must match `dtype`.

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

        Memory Support:
            - Interleaved: DRAM and L1

        Limitations:
            - Preallocated output must have the same shape as the input
            - Preallocated output for integer types is not supported

        Example:
            .. code-block:: python

                # Create tensor
                tensor_input = ttnn.rand((2,3,4), device=device)

                # Apply ttnn.cumprod() on dim=0
                tensor_output = ttnn.cumprod(tensor_input, dim=0)

                # With preallocated output and dtype
                preallocated_output = ttnn.rand([2, 3, 4], dtype=ttnn.bfloat16, device=device)

                tensor_output = ttnn.cumprod(tensor_input, dim=0, dtype=ttnn.bfloat16, out=preallocated_output)

        )doc";

    using OperationType = decltype(ttnn::cumprod);
    bind_registered_operation(
        module,
        ttnn::cumprod,
        docstring,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               std::optional<DataType>& dtype,
               const bool& reverse_order,
               std::optional<Tensor> optional_out,
               const std::optional<MemoryConfig>& memory_config) -> Tensor {
                return self(input_tensor, dim, dtype, reverse_order, optional_out, memory_config);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("reverse_order") = false,
            py::arg("out") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::accumulation::detail
