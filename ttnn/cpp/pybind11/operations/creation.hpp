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
        R"doc({0}(shape: ttnn.Shape, fill_value: Union[int, float], dtype: Optional[ttnn.DataType] = None, layout: Optional[ttnn.Layout] = None, device: Optional[ttnn.Device] = None, memory_config: Optional[ttnn.MemoryConfig] = None)doc",
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
void bind_full_operation_with_hard_coded_value(py::module& module, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(shape: ttnn.Shape, dtype: Optional[ttnn.DataType] = None, layout: Optional[ttnn.Layout] = None, device: Optional[ttnn.Device] = None, memory_config: Optional[ttnn.MemoryConfig] = None)doc",
        operation.base_name());

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
        R"doc({0}(tensor: ttnn.Tensor, fill_value: Union[int, float], dtype: Optional[ttnn.DataType] = None, layout: Optional[ttnn.Layout] = None, device: Optional[ttnn.Device] = None, memory_config: Optional[ttnn.MemoryConfig] = None)doc",
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
void bind_full_like_operation_with_hard_coded_value(py::module& module, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(tensor: ttnn.Tensor, dtype: Optional[ttnn.DataType] = None, layout: Optional[ttnn.Layout] = None, device: Optional[ttnn.Device] = None, memory_config: Optional[ttnn.MemoryConfig] = None)doc",
        operation.base_name());

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
        R"doc({0}(start: int = 0, end: int, step: int = 1, dtype: ttnn.DataType = ttnn.bfloat16, device: ttnn.Device = None, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG)doc",
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
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::zeros);
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::ones);

    detail::bind_full_like_operation(module, ttnn::full_like);
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::zeros_like);
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::ones_like);

    detail::bind_arange_operation(module, ttnn::arange);

    detail::bind_empty_operation(module);
    detail::bind_empty_like_operation(module);
}

}  // namespace creation
}  // namespace operations
}  // namespace ttnn
