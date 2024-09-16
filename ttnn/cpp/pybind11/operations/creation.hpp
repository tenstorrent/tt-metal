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
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(ttnn::Shape{tt::tt_metal::LegacyShape{shape}}, fill_value, dtype, layout, device, memory_config);
            },
            py::arg("shape"),
            py::arg("fill_value"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt},
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const std::vector<uint32_t>& shape,
               const int fill_value,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(ttnn::Shape{tt::tt_metal::LegacyShape{shape}}, fill_value, dtype, layout, device, memory_config);
            },
            py::arg("shape"),
            py::arg("fill_value"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
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
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(tensor, fill_value, dtype, layout, device, memory_config);
            },
            py::arg("tensor"),
            py::arg("fill_value"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt},
        ttnn::pybind_overload_t{
            [](const creation_operation_t& self,
               const ttnn::Tensor& tensor,
               const int fill_value,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<Device>>& device,
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(tensor, fill_value, dtype, layout, device, memory_config);
            },
            py::arg("tensor"),
            py::arg("fill_value"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
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
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(tensor, dtype, layout, device, memory_config);
            },
            py::arg("tensor"),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
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
}  // namespace creation
}  // namespace detail

void py_module(py::module& module) {
    detail::bind_full_operation(module, ttnn::full);
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::zeros);
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::ones);
    detail::bind_full_operation_with_hard_coded_value(module, ttnn::empty);

    detail::bind_full_like_operation(module, ttnn::full_like);
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::zeros_like);
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::ones_like);
    detail::bind_full_like_operation_with_hard_coded_value(module, ttnn::empty_like);

    detail::bind_arange_operation(module, ttnn::arange);
}

}  // namespace creation
}  // namespace operations
}  // namespace ttnn
