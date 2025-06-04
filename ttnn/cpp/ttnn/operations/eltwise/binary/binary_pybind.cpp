// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "ttnn-pybind/export_enum.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace binary {

namespace detail {

template <typename binary_operation_t>
void bind_primitive_binary_operation(
    py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Supports broadcasting (except with scalar)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor
            * :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor
            * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
            * :attr:`activations` (Optional[List[str]]): list of activation functions to apply to the output tensor
            * :attr:`queue_id` (Optional[uint8]): command queue id

        Example:

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
            >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               BinaryOpType binary_op_type,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<unary::FusedActivations>& activations,
               const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    binary_op_type,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activation);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("binary_op_type"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = std::nullopt,
            py::arg("input_tensor_a_activation") = std::nullopt});
}

template <typename binary_operation_t>
void bind_binary_operation(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& info = ". ",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            activations (List[str], optional): list of activation functions to apply to the output tensor{4}Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Supports broadcasting.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {5}
                 - TILE
                 - 2, 3, 4

            {6}

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        info,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    scalar,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename binary_operation_t>
void bind_binary_gcd_lcm_operation(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            activations (List[str], optional): list of activation functions to apply to the output tensor{4}Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - {5}

            {8}

        Example:
            >>> tensor1 = {6}
            >>> tensor2 = {7}
            >>> output = {1}(tensor1, tensor2)
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename binary_operation_t>
void bind_binary_unary_max_operation(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16, FLOAT32, INT32",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            activations (List[str], optional): list of activation functions to apply to the output tensor{4}Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.


        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - 2, 3, 4

            {4}

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    scalar,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename binary_operation_t>
void bind_binary_unary_operation(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& info = ". ",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            activations (List[str], optional): list of activation functions to apply to the output tensor{4}Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.


        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {5}
                 - TILE
                 - 2, 3, 4

            {6}

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        info,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    scalar,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename binary_operation_t>
void bind_bitwise_binary_ops_operation(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& info = ". ",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Integer): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.


        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {5}
                 - TILE
                 - 2, 3, 4

            {6}

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        info,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const int32_t scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    scalar,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename binary_operation_t>
void bind_binary_composite(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - {5}

            {8}

        Example:
            >>> tensor1 = {6}
            >>> tensor2 = {7}
            >>> output = {1}(tensor1, tensor2)
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_binary_composite_with_alpha(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            alpha (float): the value to be multiplied.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - 2, 3, 4

            {5}

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> alpha = 1.0
            >>> output = {1}(tensor1, tensor2, alpha)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               float alpha,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, alpha, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("alpha") = 1.0f,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_binary_composite_with_rtol_atol(
    py::module& module, const binary_operation_t& operation, const std::string& description, const std::string& math) {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            rtol (float): relative tolerance. Defaults to `1e-05f`.
            atol (float): absolute tolerance. Defaults to `1e-08f`.
            equal_nan (bool): if NaN values should be treated as equal during comparison. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

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

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> rtol = 1e-4
            >>> atol = 1e-5
            >>> equal_nan = False
            >>> output = {1}(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               float rtol,
               float atol,
               const bool equal_nan,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("rtol") = 1e-05f,
            py::arg("atol") = 1e-08f,
            py::arg("equal_nan") = false,
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_binary_composite_overload(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{output\_tensor}} = \verb|{0}|(\mathrm{{input\_tensor\_a,input\_tensor\_b}})

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - {4}

            {7}

        Example:
            >>> tensor1 = {5}
            >>> tensor2 = {6}
            >>> output = {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               const std::optional<MemoryConfig>& memory_config) { return self(input_tensor_a, value, memory_config); },
            py::arg("input_tensor_a"),
            py::arg("value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_prelu(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{output\_tensor}} = \verb|{0}|(\mathrm{{input\_tensor\_a,input\_tensor\_b}})

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or List[float] of length 1 or Number): weight.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - {4}

            {7}

        Example:
            >>> tensor1 = {5}
            >>> tensor2 = {6}
            >>> output = {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("weight"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               const std::optional<MemoryConfig>& memory_config) { return self(input_tensor_a, value, memory_config); },
            py::arg("input_tensor_a"),
            py::arg("weight"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const std::array<float, 1>& weight,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, weight, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("weight"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_div(
    py::module& module, const binary_operation_t& operation, const std::string& description, const std::string& math) {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            accurate_mode (bool, optional): `false` if input_tensor_b is non-zero, else `true` (Only if the input tensor is not ComplexTensor). Defaults to `false`.
            round_mode (string, optional): can be `None`, `floor` and `trunc` (only if the input tensor is not ComplexTensor). Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

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

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2, accurate_mode = false, round_mode = None)

            >>> tensor = ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scalar = 3
            >>> output = {1}(tensor, scalar, round_mode = "floor")
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               bool accurate_mode,
               const std::optional<std::string> round_mode,
               const std::optional<const DataType>& dtype,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    accurate_mode,
                    round_mode,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("accurate_mode") = false,
            py::arg("round_mode") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               bool accurate_mode,
               const std::optional<std::string> round_mode,
               const std::optional<const DataType>& dtype,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    value,
                    accurate_mode,
                    round_mode,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor_a"),
            py::arg("value"),
            py::kw_only(),
            py::arg("accurate_mode") = false,
            py::arg("round_mode") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename binary_operation_t>
void bind_polyval(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            Coeffs (Vector of floats): coefficients of the polynomial.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - 2, 3, 4

            {5}

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> coeffs = [1, 2, 3, 4]
            >>> output = {1}(tensor, coeffs)

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const std::vector<float>& coeffs,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, coeffs, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("coeffs"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_binary_overload_operation(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - 2, 3, 4

            {5}

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2/scalar)

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,

        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               float scalar,
               const std::optional<MemoryConfig>& memory_config) { return self(input_tensor, scalar, memory_config); },
            py::arg("input_tensor"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("input_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_inplace_operation(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                  - Ranks
                * - {4}
                  - TILE
                  - 2, 3, 4

            {5}

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,

        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               const float scalar,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    scalar,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_a"),
            py::arg("input_b"),
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename binary_operation_t>
void bind_logical_inplace_operation(
    py::module& module,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - 2, 3, 4

            {5}

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[2, 2], [2, 2]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 1], [1, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,

        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_a"),
            py::arg("input_b"),
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename binary_operation_t>
void bind_binary_inplace_operation(
    py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_a` (ttnn.Tensor)
                * :attr:`input_b` (ttnn.Tensor or Number)
            Keyword args:
            * :attr:`activations` (Optional[List[str]]): list of activation functions to apply to the output tensor
            Example::
                >>> tensor = ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,

        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               const float scalar,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    scalar,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename binary_operation_t>
void bind_power(py::module& module, const binary_operation_t& operation, const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        Perform element-wise {0} operation on :attr:`input_tensor` with :attr:`exponent`.

        .. math::
            \mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor}}_i ** \mathrm{{exponent}}_i)

        Args:
            input_tensor (ttnn.Tensor, float): the input tensor.
            exponent (float, int, ttnn.Tensor): the exponent value.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16, BFLOAT8_B
                 - TILE
                 - 2, 3, 4

            {2}

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> exponent = 2
            >>> output = {1}(tensor, exponent)
        )doc",
        ttnn::pow.base_name(),
        ttnn::pow.python_fully_qualified_name(),
        note);

    bind_registered_operation(
        module,
        ttnn::pow,
        doc,
        // integer exponent
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               uint32_t exponent,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               const QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, exponent, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg("exponent"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        // float exponent
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               float exponent,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               const QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, exponent, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg("exponent"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // tensor exponent
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               const Tensor& exponent,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    exponent,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input_tensor"),
            py::arg("exponent"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // scalar input - tensor exponent
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               float input,
               const Tensor& exponent,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input,
                    exponent,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            py::arg("input"),
            py::arg("exponent"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("use_legacy") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}
}  // namespace detail

void py_module(py::module& module) {
    export_enum<BinaryOpType>(module, "BinaryOpType");

    detail::bind_binary_operation(
        module,
        ttnn::add,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32 (range: [0, 4294967295]), UINT16 (range: [0, 65535]))doc");

    detail::bind_binary_inplace_operation(
        module,
        ttnn::add_,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::subtract,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT16 (range: 0 - 65535))doc");

    detail::bind_binary_inplace_operation(
        module,
        ttnn::subtract_,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::multiply,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i * \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, UINT16 (range: 0 - 65535))doc");

    detail::bind_binary_inplace_operation(
        module,
        ttnn::multiply_,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i \times \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::eq,
        R"doc(Compares if :attr:`input_tensor_a` is equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i == \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");

    detail::bind_binary_operation(
        module,
        ttnn::ne,
        R"doc(Compares if :attr:`input_tensor_a` is not equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i != \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");

    detail::bind_binary_operation(
        module,
        ttnn::lt,
        R"doc(Compares if :attr:`input_tensor_a` is less than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i < \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation(
        module,
        ttnn::le,
        R"doc(Compares if :attr:`input_tensor_a` is less than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i <= \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation(
        module,
        ttnn::gt,
        R"doc(Compares if :attr:`input_tensor_a` is greater than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i > \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation(
        module,
        ttnn::ge,
        R"doc(Compares if :attr:`input_tensor_a` is greater than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i >= \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation(
        module,
        ttnn::logical_and,
        R"doc(Computes logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i \, \& \, \mathrm{{input\_tensor\_b}}_i)doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        "INT32 for tensor-scalar is supported only when use_legacy= False.");

    detail::bind_binary_operation(
        module,
        ttnn::logical_or,
        R"doc(Computes logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i \, | \, \mathrm{{input\_tensor\_b}}_i)doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");

    detail::bind_binary_operation(
        module,
        ttnn::ldexp,
        R"doc(Computes ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|ldexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logaddexp,
        R"doc(Computes logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|logaddexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logaddexp2,
        R"doc(Computes logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|logaddexp2|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        module,
        ttnn::squared_difference,
        R"doc(Computes squared difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|squared_difference|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        module,
        ttnn::bias_gelu,
        R"doc(Computes bias_gelu of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|bias_gelu|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        module,
        ttnn::divide,
        R"doc(Divides :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i / \mathrm{{input\_tensor\_b}}_i))doc");

    detail::bind_binary_unary_operation(
        module,
        ttnn::rsub,
        R"doc(Subtracts :attr:`input_tensor_a` from :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_b}}_i - \mathrm{{input\_tensor\_a}}_i)doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_bitwise_binary_ops_operation(
        module,
        ttnn::bitwise_and,
        R"doc(Perform bitwise_and operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535))doc");

    detail::bind_bitwise_binary_ops_operation(
        module,
        ttnn::bitwise_or,
        R"doc(Perform bitwise_or operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_or|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535))doc");

    detail::bind_bitwise_binary_ops_operation(
        module,
        ttnn::bitwise_xor,
        R"doc(Perform bitwise_xor operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_xor|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535))doc");

    detail::bind_bitwise_binary_ops_operation(
        module,
        ttnn::bitwise_left_shift,
        R"doc(Perform bitwise_left_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31))doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32)doc");

    detail::bind_bitwise_binary_ops_operation(
        module,
        ttnn::bitwise_right_shift,
        R"doc(Perform bitwise_right_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31))doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32)doc");

    auto prim_module = module.def_submodule("prim", "Primitive binary operations");

    detail::bind_primitive_binary_operation(
        prim_module,
        ttnn::prim::binary,
        R"doc(Applied binary operation on :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    // new imported
    detail::bind_binary_composite(
        module,
        ttnn::hypot,
        R"doc(Computes hypot :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \sqrt{(\mathrm{input\_tensor\_a}_i^2 + \mathrm{input\_tensor\_b}_i^2)}
        )doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_composite(
        module,
        ttnn::xlogy,
        R"doc(Computes xlogy :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \mathrm{input\_tensor\_a}_i \cdot \log(\mathrm{input\_tensor\_b}_i)
        )doc");

    detail::bind_binary_composite(
        module,
        ttnn::nextafter,
        R"doc(Computes nextafter :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \begin{cases} \mathrm{next\_float}(\mathrm{input\_tensor\_a}_i, \mathrm{input\_tensor\_b}_i), & \text{if } \mathrm{input\_tensor\_a}_i \neq \mathrm{input\_tensor\_b}_i \\ \mathrm{input\_tensor\_a}_i, & \text{if } \mathrm{input\_tensor\_a}_i = \mathrm{input\_tensor\_b}_i \end{cases}
        )doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_unary_max_operation(
        module,
        ttnn::minimum,
        R"doc(Computes minimum for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_binary_composite(
        module,
        ttnn::atan2,
        R"doc(Computes atan2 :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \arctan\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)
        )doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(2, 3, 4)doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(Input arguments for the atan2 function are in the format (y, x))doc");

    detail::bind_binary_operation(
        module,
        ttnn::logical_xor,
        R"doc(Compute logical_xor :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = (\mathrm{input\_tensor\_a}_i \land \lnot \mathrm{input\_tensor\_b}_i) \lor (\lnot \mathrm{input\_tensor\_a}_i \land \mathrm{input\_tensor\_b}_i))doc",
        ".",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");

    detail::bind_logical_inplace_operation(
        module,
        ttnn::logical_or_,
        R"doc(Computes inplace logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i | \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");

    detail::bind_logical_inplace_operation(
        module,
        ttnn::logical_xor_,
        R"doc(Computes inplace logical XOR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{input\_tensor\_a}_i \land \lnot \mathrm{input\_tensor\_b}_i) \lor (\lnot \mathrm{input\_tensor\_a}_i \land \mathrm{input\_tensor\_b}_i)doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");

    detail::bind_logical_inplace_operation(
        module,
        ttnn::logical_and_,
        R"doc(Computes inplace logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i \& \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_gcd_lcm_operation(
        module,
        ttnn::gcd,
        R"doc(Computes Greatest common divisor of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`.
        [supported range [-2147483647, 2147483648]].)doc",
        R"doc(\mathrm{output\_tensor}_i = \verb|gcd|\left(\mathrm{input\_tensor\_a}_i , \mathrm{input\_tensor\_b}_i\right)
        )doc",
        R"doc(INT32)doc",
        R"doc(2, 3, 4)doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))doc");

    detail::bind_binary_gcd_lcm_operation(
        module,
        ttnn::lcm,
        R"doc(Computes Least common multiple of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`.
        [supported range [-32767, 32768]].)doc",
        R"doc(\mathrm{output\_tensor}_i = \verb|lcm|\left(\mathrm{input\_tensor\_a}_i , \mathrm{input\_tensor\_b}_i\right)
        )doc",
        R"doc(INT32)doc",
        R"doc(2, 3, 4)doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))doc");

    detail::bind_binary_composite_with_alpha(
        module,
        ttnn::addalpha,
        R"doc(Computes addalpha for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`.)doc",
        R"doc(\mathrm{{output\_tensor}} = \mathrm{{input\_tensor\_a\ + input\_tensor\_b\ * \alpha}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_composite_with_alpha(
        module,
        ttnn::subalpha,
        R"doc(Computes subalpha for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \mathrm{{input\_tensor\_a\ - input\_tensor\_b\ * \alpha}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_composite_with_rtol_atol(
        module,
        ttnn::isclose,
        R"doc(Computes isclose for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \begin{cases} 1, & \text{if } |\mathrm{input\_tensor\_a} - \mathrm{input\_tensor\_b}| \leq (\mathrm{atol} + \mathrm{rtol} \times |\mathrm{input\_tensor\_b}|) \\ 0, & \text{otherwise} \end{cases}
        )doc");

    detail::bind_div(
        module,
        ttnn::div,
        R"doc(Computes div for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output}_i = \begin{cases} \mathrm{\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{None} \\ \mathrm{\text{floor}\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{floor} \\ \mathrm{\text{trunc}\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{trunc} \end{cases}
        )doc");

    detail::bind_binary_composite_overload(
        module,
        ttnn::div_no_nan,
        R"doc(Computes div_no_nan for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_binary_composite_overload(
        module,
        ttnn::floor_div,
        R"doc(Computes floor division for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_binary_unary_max_operation(
        module,
        ttnn::maximum,
        R"doc(Computes maximum for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_prelu(
        module,
        ttnn::prelu,
        R"doc(Perform an eltwise-prelu operation.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(2, 3, 4, 5)doc",
        R"doc(ttnn.from_torch(torch.rand([1, 2, 32, 32], dtype=torch.bfloat16), device=device))doc",
        R"doc(ttnn.from_torch(torch.tensor([1, 2], dtype=torch.bfloat16), device=device))doc",
        R"doc(PReLU supports the case where weight is a scalar or 1D list/array of size=1 or a 1D tensor :attr:`input_tensor_b` of size = the second dimension in :attr:`input_tensor_a`)doc");

    detail::bind_binary_composite(
        module,
        ttnn::scatter,
        R"doc(Computes scatter for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output}_i = \verb|scatter|\left(\mathrm{input\_tensor\_a}_i , \mathrm{input\_tensor\_b}_i\right))doc",
        R"doc(BFLOAT16)doc",
        R"doc(4)doc",
        R"doc(ttnn.from_torch(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(ttnn.from_torch(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device))doc");

    detail::bind_binary_composite(
        module,
        ttnn::outer,
        R"doc(Computes outer for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \mathrm{input\_tensor\_a} \text{ } \otimes \text{ } \mathrm{input\_tensor\_b})doc",
        R"doc(BFLOAT16)doc",
        R"doc(4)doc",
        R"doc(ttnn.from_torch(torch.rand([1, 1, 32, 1], dtype=torch.bfloat16), device=device))doc",
        R"doc(ttnn.from_torch(torch.rand([1, 1, 1, 32], dtype=torch.bfloat16), device=device))doc");

    detail::bind_polyval(
        module,
        ttnn::polyval,
        R"doc(Computes polyval of all elements of :attr:`input_tensor_a` with coefficients :attr:`coeffs` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \sum_{i=0}^{n} (\mathrm{coeffs}_i) (\mathrm{input\_tensor}^i)
        )doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_overload_operation(
        module,
        ttnn::fmod,
        R"doc(Performs an eltwise-fmod operation.)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|fmod|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, FLOAT32)doc");

    detail::bind_binary_overload_operation(
        module,
        ttnn::remainder,
        R"doc(Performs an eltwise-modulus operation.)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|remainder|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::gt_,
        R"doc(Performs Greater than in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} > \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::ge_,
        R"doc(Performs Greater than or equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} >= \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::lt_,
        R"doc(Performs Less than in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} < \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::le_,
        R"doc(Performs Less than or equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} <= \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::eq_,
        R"doc(Performs Equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} == \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::ne_,
        R"doc(Performs Not equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}\: != \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::ldexp_,
        R"doc(Performs ldexp in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|ldexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::logaddexp_,
        R"doc(Performs logaddexp in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|logaddexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::logaddexp2_,
        R"doc(Performs logaddexp2 in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|logaddexp2|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::squared_difference_,
        R"doc(Performs squared_difference in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|squared_difference|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::divide_,
        R"doc(Performs division in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|squared_difference|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc");

    detail::bind_inplace_operation(
        module,
        ttnn::rsub_,
        R"doc(Subtracts :attr:`input_a` from :attr:`input_b` in-place and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_b}} - \mathrm{{input\_tensor\_a}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::bias_gelu_,
        R"doc(Performs bias_gelu in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|bias_gelu|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_power(
        module,
        ttnn::pow,
        R"doc(When :attr:`exponent` is a Tensor, supported dtypes are: BFLOAT16, FLOAT32. Both input tensors should be of same dtype.)doc");
}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
