// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_pybind.hpp"

#include "ttnn/operations/reduction/topk/topk.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::reduction::detail {
namespace py = pybind11;

void bind_reduction_topk_operation(py::module& module) {
    auto doc =
        R"doc(topk(input_tensor: ttnn.Tensor, k: int, dim: int, largest: bool, sorted: bool, out : Optional[ttnn.Tensor] = std::nullopt, memory_config: MemoryConfig = std::nullopt, ) -> Tuple[ttnn.Tensor, ttnn.Tensor]

            Returns the :attr:`k` largest or :attr:`k` smallest elements of the :attr:`input_tensor` along a given dimension :attr:`dim`.

            If :attr:`dim` is not provided, the last dimension of the :attr:`input_tensor` is used.

            If :attr:`largest` is True, the :attr:`k` largest elements are returned. Otherwise, the :attr:`k` smallest elements are returned.

            The boolean option :attr:`sorted` if True, will make sure that the returned :attr:`k` elements are sorted.

            Equivalent PyTorch code:

            .. code-block:: python

                return torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=sorted, *, out=None)

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                k (number): the number of top elements to look for.
                dim (number): the dimension to reduce.
                largest (bool): whether to return the largest or the smallest elements. Defaults to `False`.
                sorted (bool): whether to return the elements in sorted order. Defaults to `False`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Core range set to run the operation on. Defaults to `None`.
                indices_tensor (ttnn.Tensor, optional): Preallocated indices tensor. Defaults to `None`.

            Returns:
                List of ttnn.Tensor: the output tensor.

            Note:
                The :attr:`input_tensor` supports the following data type and layout:

                .. list-table:: input_tensor
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - BFLOAT8, BFLOAT16
                        - TILE

                .. list-table:: index_tensor
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - UINT16, UINT32
                        - TILE

                The :attr:`output_value_tensor` will have the same data type as :attr:`input_tensor` and :attr:`output_index_tensor` will have UINT16 data type.

            Limitations:
                - The op fundamentally operates on 4D tensors with shape [N, C, H, W], and with :attr:`dim` of -1. The tensor will be manipulated as needed when this is not the case, and restored afterwards.
                - For :attr:`input_tensor`, N*C*H must be a multiple of 32
                - W is ideally ≥64. If this is not the case the op will pad the tensor to satisfy this constraint.
                - The width of :attr:`input_tensor` along :attr:`dim` should be a multiple of tile width, and will be padded to the nearest multiple of tile width if needed.
                - The padding is currently only supported for bfloat16, float32, int32, and uint32.
                - To enable multicore execution, the width of :attr:`input_tensor` along :attr:`dim` must be ≥8192 and <65536, and :attr:`k` must be ≤64.
                - All shape validations are performed on padded shapes.

            Example:
                input_tensor = ttnn.rand([1, 1, 32, 64], device=device, layout=ttnn.TILE_LAYOUT)
                topk_values, topk_indices = ttnn.topk(input_tensor, k=32, dim=-1, largest=True, sorted=True)

        )doc";

    using OperationType = decltype(ttnn::topk);
    bind_registered_operation(
        module,
        ttnn::topk,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t k,
               const int8_t dim,
               const bool largest,
               const bool sorted,
               std::optional<std::tuple<ttnn::Tensor, ttnn::Tensor>> optional_output_tensors,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::CoreRangeSet>& sub_core_grids,
               const std::optional<ttnn::Tensor>& indices_tensor) {
                return self(
                    input_tensor,
                    k,
                    dim,
                    largest,
                    sorted,
                    memory_config,
                    sub_core_grids,
                    indices_tensor,
                    optional_output_tensors);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("k") = 32,
            py::arg("dim") = -1,
            py::arg("largest") = true,
            py::arg("sorted") = true,
            py::kw_only(),
            py::arg("out") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt,
            py::arg("indices_tensor") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::detail
