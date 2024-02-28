// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tt_eager/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace types {

void py_module(py::module& module) {
    // TODO: @eyonland, figure out how to migrate these duplicate pybindings.

    // py::class_<ttnn::types::MemoryConfig>(module, "MemoryConfig")
    //     .def(py::init<>())
    //     .def_readwrite("memory_layout", &ttnn::types::MemoryConfig::memory_layout)
    //     .def_readwrite("buffer_type", &ttnn::types::MemoryConfig::buffer_type);

    // py::enum_<tt::tt_metal::TensorMemoryLayout>(module, "TensorMemoryLayout")
    //     .value("INTERLEAVED", tt::tt_metal::TensorMemoryLayout::INTERLEAVED)
    //     .export_values();

    // py::class_<tt::tt_metal::Device>(module, "Device");

    module.attr("DRAM_MEMORY_CONFIG") = py::cast(DRAM_MEMORY_CONFIG);
    module.attr("L1_MEMORY_CONFIG") = py::cast(L1_MEMORY_CONFIG);

    auto PyShape = py::class_<ttnn::Shape>(module, "Shape");
    PyShape.def(py::init<tt::tt_metal::Shape>());

    [&PyShape]<auto... Ns>(std::index_sequence<Ns...>) {
        (
            [&PyShape]() {
                if constexpr (Ns > 0) {
                    PyShape.def(py::init<const std::array<uint32_t, Ns>&>(), py::arg("shape"));

                    PyShape.def(
                        py::init<const std::array<uint32_t, Ns>&, const std::array<std::array<uint32_t, 2>, Ns>&>(),
                        py::arg("shape"),
                        py::arg("padding"));

                    PyShape.def(
                        py::init<const std::array<uint32_t, Ns>&, std::array<uint32_t, Ns>&>(),
                        py::arg("shape"),
                        py::arg("padded_shape"));

                    PyShape.def("__eq__", [](const Shape& self, const std::array<uint32_t, Ns>& other) {
                        return Shape{self.value().without_padding()} == Shape{tt::tt_metal::Shape{other}};
                    });
                }
            }(),
            ...);
    }(std::make_index_sequence<8>());

    PyShape.def_property_readonly("value", [](const Shape& self) { return self.value(); });
    PyShape.def("__len__", [](const Shape& self) { return self.rank(); });
    PyShape.def("__getitem__", [](const Shape& self, std::int64_t index) { return self[index]; });
    PyShape.def("__iter__", [](const Shape& self) { return py::iter(py::cast(self.value().without_padding())); });
    PyShape.def("__eq__", [](const Shape& self, const Shape& other) { return self == other; });
    PyShape.def("__eq__", [](const Shape& self, const py::none) { return false; });
    PyShape.def("__repr__", [](const Shape& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });
    PyShape.def_property_readonly("rank", [](const Shape& self) -> std::size_t { return self.rank(); });
    PyShape.def("with_tile_padding", [](const Shape& self) { return self.with_tile_padding(); });

    py::class_<TensorWrapper>(module, "Tensor")
        .def(py::init<tt::tt_metal::Tensor>())
        .def_property_readonly("value", [](const TensorWrapper& self) -> auto& { return self.value; })
        .def(
            "__repr__",
            [](const TensorWrapper& self) {
                if (self.value.is_allocated()) {
                    return self.value.write_to_string(Layout::ROW_MAJOR, true);
                } else {
                    return fmt::format(
                        "Tensor(shape={}, dtype={}, layout={})",
                        self.value.ttnn_shape(),
                        self.value.dtype(),
                        self.value.layout());
                }
            })
        .def_property_readonly("shape", [](const TensorWrapper& self) { return py::cast(Shape{self.value.shape()}); })
        .def_property_readonly("dtype", [](const TensorWrapper& self) { return self.value.dtype(); })
        .def_property_readonly("layout", [](const TensorWrapper& self) { return self.value.layout(); })
        .def_property_readonly("device", [](const TensorWrapper& self) -> Device& { return *self.value.device(); })
        .def("is_contiguous", [](const TensorWrapper& self) -> bool { return self.value.is_contiguous(); });
}

}  // namespace types
}  // namespace ttnn
