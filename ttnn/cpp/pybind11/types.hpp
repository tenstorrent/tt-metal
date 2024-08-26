// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "export_enum.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace types {

void py_module_types(py::module& module) {
    py::class_<ttnn::CoreGrid>(module, "CoreGrid");
    py::class_<ttnn::Shape>(module, "Shape");

    module.attr("DRAM_MEMORY_CONFIG") = py::cast(DRAM_MEMORY_CONFIG);
    module.attr("L1_MEMORY_CONFIG") = py::cast(L1_MEMORY_CONFIG);
}

void py_module(py::module& module) {
    auto py_core_coord = static_cast<py::class_<ttnn::CoreGrid>>(module.attr("CoreGrid"));
    py_core_coord.def(py::init<std::size_t, std::size_t>(), py::kw_only(), py::arg("x"), py::arg("y"))
        .def_property_readonly("x", [](const ttnn::CoreGrid& self) { return self.x; })
        .def_property_readonly("y", [](const ttnn::CoreGrid& self) { return self.y; })
        .def_property_readonly("num_cores", [](const ttnn::CoreGrid& self) { return self.x * self.y; })
        .def("__repr__", [](const ttnn::CoreGrid& self) -> std::string {
            std::stringstream ss;
            ss << self;
            return ss.str();
        });

    auto PyShape = static_cast<py::class_<ttnn::Shape>>(module.attr("Shape"));
    PyShape.def(py::init<tt::tt_metal::Shape>())
        .def_property_readonly("value", [](const Shape& self) { return self.value; })
        .def("__len__", [](const Shape& self) { return self.rank(); })
        .def("__getitem__", [](const Shape& self, std::int64_t index) { return self[index]; })
        .def("__iter__", [](const Shape& self) { return py::iter(py::cast(self.value.without_padding())); })
        .def(pybind11::self == pybind11::self)
        .def(
            "__repr__",
            [](const Shape& self) {
                std::stringstream ss;
                ss << self;
                return ss.str();
            })
        .def_property_readonly("rank", [](const Shape& self) -> std::size_t { return self.rank(); })
        .def("with_tile_padding", [](const Shape& self) { return self.with_tile_padding(); });

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

                    PyShape.def(pybind11::self == std::array<uint32_t, Ns>{});
                }
            }(),
            ...);
    }(std::make_index_sequence<8>());
}

}  // namespace types
}  // namespace ttnn
