// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor/tensor.hpp"

namespace py = pybind11;

namespace ttnn {
namespace types {

void py_module(py::module& m_types) {
    auto PyShape = py::class_<Shape>(m_types, "Shape");
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

    struct Tensor {
        tt::tt_metal::Tensor value;
    };

    py::class_<Tensor>(m_types, "Tensor")
        .def(py::init<tt::tt_metal::Tensor>())
        .def_property_readonly("value", [](const Tensor& self) -> auto& { return self.value; })
        .def("__repr__", [](const Tensor& self) { return self.value.write_to_string(Layout::ROW_MAJOR, true); })
        .def_property_readonly("shape", [](const Tensor& self) { return py::cast(Shape{self.value.shape()}); })
        .def_property_readonly("dtype", [](const Tensor& self) { return self.value.dtype(); })
        .def_property_readonly("layout", [](const Tensor& self) { return self.value.layout(); })
        .def_property_readonly(
            "device",
            [](const Tensor& self) -> Device* {
                if (self.value.storage_type() == tt::tt_metal::StorageType::DEVICE) {
                    return self.value.device();
                } else {
                    throw std::runtime_error("Tensor is not on device!");
                }
            })
        .def("is_contiguous", [](const Tensor& self) -> bool {
            if (self.value.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
                return self.value.shape() == self.value.shape().without_padding();
            } else {
                return false;
            }
        });
}

}  // namespace types
}  // namespace ttnn
