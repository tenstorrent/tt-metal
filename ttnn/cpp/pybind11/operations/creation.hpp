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
        Creates a tensor of the specified shape and fills it with the specified scalar value.

        Args:
            shape (ttnn.Shape): The shape of the tensor.
            fill_value (float): The value to fill the tensor with.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Note:
            ROW_MAJOR_LAYOUT requires last dimension (shape[-1]) to be a multiple of 2 with dtype BFLOAT16 or UINT16.
            TILE_LAYOUT requires width (shape[-1]) and height (shape[-2]) dimension to be multiple of 32.

        Returns:
            ttnn.Tensor: A filled tensor of specified shape and value.

        Example:
            >>> filled_tensor = ttnn.full(shape=[2, 2], fill_value=7.0, dtype=ttnn.bfloat16)
            >>> print(filled_tensor)
            ttnn.Tensor([[[[7.0,  7.0],
                            [7.0,  7.0]]]], shape=Shape([2, 2]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name());

    bind_registered_operation(module,
                              operation,
                              doc,
                              ttnn::pybind_overload_t{[](const creation_operation_t& self,
                                                         const std::vector<uint32_t>& shape,
                                                         const float fill_value,
                                                         const std::optional<DataType>& dtype,
                                                         const std::optional<Layout>& layout,
                                                         const std::optional<std::reference_wrapper<Device>>& device,
                                                         const std::optional<MemoryConfig>& memory_config,
                                                         std::optional<ttnn::Tensor>& optional_output_tensor,
                                                         uint8_t queue_id) -> ttnn::Tensor {
                                                          return self(queue_id,
                                                                      ttnn::Shape{tt::tt_metal::LegacyShape{shape}},
                                                                      fill_value,
                                                                      dtype,
                                                                      layout,
                                                                      device,
                                                                      memory_config,
                                                                      optional_output_tensor);
                                                      },
                                                      py::arg("shape"),
                                                      py::arg("fill_value"),
                                                      py::arg("dtype") = std::nullopt,
                                                      py::arg("layout") = std::nullopt,
                                                      py::arg("device") = std::nullopt,
                                                      py::arg("memory_config") = std::nullopt,
                                                      py::arg("optional_tensor") = std::nullopt,
                                                      py::arg("queue_id") = ttnn::DefaultQueueId},
                              ttnn::pybind_overload_t{[](const creation_operation_t& self,
                                                         const std::vector<uint32_t>& shape,
                                                         const int fill_value,
                                                         const std::optional<DataType>& dtype,
                                                         const std::optional<Layout>& layout,
                                                         const std::optional<std::reference_wrapper<Device>>& device,
                                                         const std::optional<MemoryConfig>& memory_config,
                                                         std::optional<ttnn::Tensor>& optional_output_tensor,
                                                         uint8_t queue_id) -> ttnn::Tensor {
                                                          return self(queue_id,
                                                                      ttnn::Shape{tt::tt_metal::LegacyShape{shape}},
                                                                      fill_value,
                                                                      dtype,
                                                                      layout,
                                                                      device,
                                                                      memory_config,
                                                                      optional_output_tensor);
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
void bind_full_operation_with_hard_coded_value(py::module& module,
                                               const creation_operation_t& operation,
                                               const std::string& value_string) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor with the specified shape and fills it with the value of {1}.

        Args:
            shape (ttnn.Shape): The shape of the tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.

        Note:
            ROW_MAJOR_LAYOUT requires last dimension (shape[-1]) to be a multiple of 2 with dtype BFLOAT16 or UINT16.
            TILE_LAYOUT requires requires width (shape[-1]), height (shape[-2]) dimension to be multiple of 32.

        Returns:
            ttnn.Tensor: A tensor filled with {1}.

        Example:
            >>> tensor = ttnn.{0}(shape=[1, 2, 2, 2], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            >>> print(tensor)
            ttnn.Tensor([[[[{1}, {1}],
                            [{1}, {1}]],
                            [[{1}, {1}],
                            [{1}, {1}]]]]], shape=Shape([1, 2, 2, 2]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name(),
        value_string);

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
        Creates a tensor of the same shape as the input tensor and fills it with the specified scalar value. The data type, layout, device, and memory configuration of the resulting tensor can be specified.

        Args:
            tensor (ttnn.Tensor): The tensor to use as a template for the shape of the new tensor.
            fill_value (float | int): The value to fill the tensor with.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: A filled tensor.

        Example:
            >>> tensor = ttnn.zeros(shape=(2, 3), dtype=ttnn.bfloat16)
            >>> filled_tensor = ttnn.full_like(tensor, fill_value=5.0, dtype=ttnn.bfloat16)
            >>> print(filled_tensor)
            ttnn.Tensor([[[[5.0,  5.0,  5.0],
                            [5.0,  5.0,  5.0]]]], shape=Shape([2, 3]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
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
               std::optional<ttnn::Tensor>& optional_output_tensor,
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
               std::optional<ttnn::Tensor>& optional_output_tensor,
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
void bind_full_like_operation_with_hard_coded_value(py::module& module,
                                                    const creation_operation_t& operation,
                                                    const std::string& value_string) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the same shape as the input tensor and fills it with the value of {1}. The data type, layout, device, and memory configuration of the resulting tensor can be specified.

        Args:
            tensor (ttnn.Tensor): The tensor to use as a template for the shape of the new tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: A tensor filled with {1}.

        Example:
            >>> tensor = ttnn.{0}(ttnn.from_torch(torch.randn(1, 2, 2, 2), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            >>> output_tensor = ttnn.{0}(tensor=input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            >>> print(output_tensor)
            ttnn.Tensor([[[[{1}, {1}],
                            [{1}, {1}]],
                            [[{1}, {1}],
                            [{1}, {1}]]]]], shape=Shape([1, 2, 2, 2]), dtype=DataType::BFLOAT16, layout=Layout::TILE_LAYOUT)
        )doc",
        operation.base_name(),
        value_string);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{[](const creation_operation_t& self,
                                   const ttnn::Tensor& tensor,
                                   const std::optional<DataType>& dtype,
                                   const std::optional<Layout>& layout,
                                   const std::optional<std::reference_wrapper<Device>>& device,
                                   const std::optional<MemoryConfig>& memory_config,
                                   std::optional<ttnn::Tensor>& optional_output_tensor,
                                   uint8_t queue_id) -> ttnn::Tensor {
                                    return self(
                                        queue_id, tensor, dtype, layout, device, memory_config, optional_output_tensor);
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
            ttnn.Tensor([[[[0.00000,  2.00000,  ...,  8.00000,  0.00000]]]], shape=Shape([1, 1, 1, 6]), dtype=DataType::FLOAT32, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name());

    bind_registered_operation(module,
                              operation,
                              doc,
                              ttnn::pybind_overload_t{[](const creation_operation_t& self,
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
        R"doc(
        Creates a device tensor with uninitialized values of the specified shape, data type, layout, and memory configuration.

        Args:
            shape (List[int]): The shape of the tensor to be created.
            dtype (ttnn.DataType, optional): The tensor data type. Defaults to `ttnn.bfloat16`.
            layout (ttnn.Layout, optional): The tensor layout. Defaults to `ttnn.ROW_MAJOR`.
            device (ttnn.Device): The device where the tensor will be allocated.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.

        Returns:
            ttnn.Tensor: The output uninitialized tensor.

        Example:
            >>> tensor = ttnn.empty(shape=[2, 3], device=device)
            >>> print(tensor)
            ttnn.Tensor([[[[0.9, 0.21, 0.5], [0.67, 0.11, 0.30]]]], shape=Shape([2, 3]), dtype=DataType::BFLOAT16, layout=Layout::TILE)
        )doc",
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
            py::arg("device"),
            py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG});
}

void bind_empty_like_operation(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Creates a new tensor with the same shape as the given `reference`, but without initializing its values. The data type, layout, device, and memory configuration of the new tensor can be specified.

        Args:
            reference (ttnn.Tensor): The reference tensor whose shape will be used for the output tensor.

        Keyword Args:
            dtype (ttnn.DataType, optional): The desired data type of the output tensor. Defaults to `ttnn.bfloat16`.
            layout (ttnn.Layout, optional): The desired layout of the output tensor. Defaults to `ttnn.ROW_MAJOR`.
            device (ttnn.Device, optional): The device where the output tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.

        Returns:
            ttnn.Tensor: The output uninitialized tensor with the same shape as the reference tensor.

        Example:
            >>> reference = ttnn.from_torch(torch.randn(2, 3), dtype=ttnn.bfloat16)
            >>> tensor = ttnn.empty_like(reference, dtype=ttnn.float32)
            >>> print(tensor)
            ttnn.Tensor([[[[0.87, 0.45, 0.22], [0.60, 0.75, 0.25]]]], shape=Shape([2, 3]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        ttnn::empty_like.base_name());

    using EmptyLikeType = decltype(ttnn::empty_like);
    bind_registered_operation(
        module,
        ttnn::empty_like,
        doc,
        ttnn::pybind_overload_t{[](const EmptyLikeType& self,
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
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::zeros, "0.0");
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::ones, "1.0");

    detail::bind_full_like_operation(module, ttnn::full_like);
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::zeros_like, "0.0");
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::ones_like, "1.0");

    detail::bind_arange_operation(module, ttnn::arange);

    detail::bind_empty_operation(module);
    detail::bind_empty_like_operation(module);
}

}  // namespace creation
}  // namespace operations
}  // namespace ttnn
