// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/creation.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace creation {

namespace detail {

template <typename creation_operation_t>
void bind_full_operation(py::module& module, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the specified shape and fills it with the specified value.

        Args:
            shape (ttnn.Shape): The shape of the tensor.
            fill_value (float): The value to fill the tensor with.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to None.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to None.

        Returns:
            ttnn.Tensor: A filled tensor.
        )doc",
        operation.base_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const std::vector<uint32_t>& shape,
               const float fill_value,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> &optional_output_tensor,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, ttnn::Shape{tt::tt_metal::LegacyShape{shape}}, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
            },
            py::arg("shape"),
            py::arg("fill_value"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("optional_tensor") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const std::vector<uint32_t>& shape,
               const int fill_value,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> &optional_output_tensor,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, ttnn::Shape{tt::tt_metal::LegacyShape{shape}}, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
            },
            py::arg("shape"),
            py::arg("fill_value"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("optional_tensor") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename creation_operation_t>
void bind_full_operation_with_hard_coded_value(py::module& module, const creation_operation_t& operation, const std::string& info_doc = "" ) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor with the specified shape and fills it with the value of {0}.

        Args:
            shape (ttnn.Shape): The shape of the tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to None.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to None.

        Returns:
            ttnn.Tensor: A filled tensor.

        Note:
            {2}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const std::vector<uint32_t>& shape,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(ttnn::Shape{tt::tt_metal::LegacyShape{shape}}, dtype, layout, device, memory_config);
            },
            py::arg("shape"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

template <typename creation_operation_t>
void bind_full_like_operation(py::module& module, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the same shape as the input tensor and fills it with the specified value.

        Args:
            tensor (ttnn.Tensor): The tensor to use as a template for the shape of the new tensor.
            fill_value (float): The value to fill the tensor with.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to None.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to None.

        Returns:
            ttnn.Tensor: A filled tensor.
        )doc",
        operation.base_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const ttnn::Tensor& tensor,
               const float fill_value,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> &optional_output_tensor,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
            },
            py::arg("tensor"),
            py::arg("fill_value"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("optional_tensor") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const ttnn::Tensor& tensor,
               const int fill_value,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> &optional_output_tensor,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
            },
            py::arg("tensor"),
            py::arg("fill_value"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("optional_tensor") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename creation_operation_t>
void bind_full_like_operation_with_hard_coded_value(py::module& module, const creation_operation_t& operation, const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the same shape as the input tensor and fills it with the value of {0}.

        Args:
            tensor (ttnn.Tensor): The tensor to use as a template for the shape of the new tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to None.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to None.

        Returns:
            ttnn.Tensor: A filled tensor.

        Note:
            {2}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const ttnn::Tensor& tensor,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> &optional_output_tensor,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, tensor, dtype, layout, device, memory_config, optional_output_tensor);
            },
            py::arg("tensor"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("optional_tensor") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename creation_operation_t>
void bind_arange_operation(py::module& module, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor with values ranging from `start` (inclusive) to `end` (exclusive) with a specified `step` size. The data type, device, and memory configuration of the resulting tensor can be specified.

        Args:
            start (int, optional): The start of the range. Defaults to 0.
            end (int): The end of the range (exclusive).
            step (int, optional): The step size between consecutive values. Defaults to 1.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `ttnn.bfloat16`.
            device (ttnn.Device, optional): The device where the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration for the tensor. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.

        Returns:
            ttnn.Tensor: A tensor containing evenly spaced values within the specified range.

        Example:
            >>> tensor = ttnn.arange(start=0, end=10, step=2, dtype=ttnn.float32)
            >>> print(tensor)
            ttnn.Tensor([[[[ 0.00000,  2.00000,  ...,  8.00000,  0.00000]]]], shape=Shape([1, 1, 1, 6]), dtype=DataType::FLOAT32, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const int64_t start,
               const int64_t end,
               const int64_t step,
               const DataType dtype,
               const std::optional<std::reference_wrapper<Device>>& device,
               const MemoryConfig& memory_config) -> ttnn::Tensor {
                return self(start, end, step, dtype, device, memory_config);
            },
            py::arg("start") = 0,
            py::arg("end"),
            py::arg("step") = 1,
            py::arg("dtype") = ttnn::bfloat16,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG});
}

void bind_empty_operation(py::module& module) {
    auto doc = fmt::format(
        R"doc({0}(shape: List[int], dtype: ttnn.DataType, layout: ttnn.Layout, device: ttnn.Device, memory_config: ttnn.MemoryConfig)doc",
        ttnn::empty.base_name());

    using EmptyType = decltype(ttnn::empty);
    bind_registered_operation(
        module,
        ttnn::empty,
        doc,
        ttnn::pybind_overload_t{
            [](const EmptyType& self,
               const std::vector<uint32_t>& shape,
               const DataType& dtype,
               const Layout& layout,
               Device* device,
               const MemoryConfig& memory_config) -> ttnn::Tensor {
                return self(ttnn::Shape{tt::tt_metal::LegacyShape{shape}}, dtype, layout, device, memory_config);
            },
            py::arg("shape"),
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("layout") = Layout::ROW_MAJOR,
            py::arg("device") = nullptr,
            py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG});
}

void bind_empty_like_operation(py::module& module) {
    auto doc = fmt::format(
        R"doc({0}(tensor: ttnn.Tensor, dtype: Optional[ttnn.DataType] = None, layout: Optional[ttnn.Layout] = None, device: Optional[ttnn.Device] = None, memory_config: Optional[ttnn.MemoryConfig] = None)doc",
        ttnn::empty_like.base_name());

    using EmptyLikeType = decltype(ttnn::empty_like);
    bind_registered_operation(
        module,
        ttnn::empty_like,
        doc,
        ttnn::pybind_overload_t{
            [](const EmptyLikeType& self,
               const ttnn::Tensor& reference,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(reference, dtype, layout, device, memory_config);
            },
            py::arg("tensor"),
            py::kw_only(),
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("layout") = Layout::ROW_MAJOR,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_full_operation(module, ttnn::full);
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::zeros,
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       ROW_MAJOR, TILE           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT_8                |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::ones);

    detail::bind_full_like_operation(module, ttnn::full_like);
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::zeros_like,
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       ROW_MAJOR, TILE           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::ones_like);

    detail::bind_arange_operation(module, ttnn::arange);

    detail::bind_empty_operation(module);
    detail::bind_empty_like_operation(module);
}

}  // namespace creation
}  // namespace operations
}  // namespace ttnn
