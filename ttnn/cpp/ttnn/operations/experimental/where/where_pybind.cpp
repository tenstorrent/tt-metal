// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/where/where_pybind.hpp"
#include "ttnn/operations/experimental/where/where.hpp"

namespace ttnn::operations::ternary::experimental::detail {

void bind_where(pybind11::module& module) {
    auto operation = ttnn::operations::ternary::experimental::where;
    using OperationType = decltype(ttnn::operations::ternary::experimental::where);
    std::string description =
        R"doc(Computes Where on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc";
    auto doc = fmt::format(
        R"doc(
            {2}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.


        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.


        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 0], [1, 0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor3 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2, tensor3)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& predicate,
               const Tensor& true_value,
               const Tensor& false_value,
               std::optional<const DataType> output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               QueueId queue_id) {
                return self(
                    queue_id,
                    predicate,
                    true_value,
                    false_value,
                    output_dtype,
                    memory_config,
                    std::move(output_tensor));
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            py::arg("dtype").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& predicate,
               const float true_value,
               const Tensor& false_value,
               std::optional<const DataType> output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               QueueId queue_id) {
                return self(
                    queue_id,
                    predicate,
                    true_value,
                    false_value,
                    output_dtype,
                    memory_config,
                    std::move(output_tensor));
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            pybind11::arg("dtype").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& predicate,
               const Tensor& true_value,
               const float false_value,
               std::optional<const DataType> output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               QueueId queue_id) {
                return self(
                    queue_id,
                    predicate,
                    true_value,
                    false_value,
                    output_dtype,
                    memory_config,
                    std::move(output_tensor));
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            py::arg("dtype").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& predicate,
               const float true_value,
               const float false_value,
               std::optional<const DataType> output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               QueueId queue_id) {
                return self(
                    queue_id,
                    predicate,
                    true_value,
                    false_value,
                    output_dtype,
                    memory_config,
                    std::move(output_tensor));
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            py::arg("dtype").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::ternary::experimental::detail
