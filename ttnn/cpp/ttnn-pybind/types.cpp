// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/export_enum.hpp"
#include "ttnn-pybind/small_vector_caster.hpp"  // NOLINT - for pybind11 SmallVector binding support.
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::types {

void py_module_types(py::module& module) {
    py::class_<ttnn::CoreGrid>(module, "CoreGrid");
    py::class_<ttnn::Shape>(module, "Shape");
    py::class_<ttnn::QueueId>(module, "QueueId")
        .def(py::init<uint8_t>())
        .def("__int__", [](const ttnn::QueueId& self) { return static_cast<int>(*self); })
        .def(
            "__repr__",
            [](const ttnn::QueueId& self) { return "QueueId(" + std::to_string(static_cast<int>(*self)) + ")"; })
        .def(py::self == py::self);

    export_enum<ttnn::BcastOpMath>(module, "BcastOpMath");
    export_enum<ttnn::BcastOpDim>(module, "BcastOpDim");

    py::implicitly_convertible<py::int_, ttnn::QueueId>();

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
    PyShape.def(py::init<const ttnn::SmallVector<uint32_t>&>(), py::arg("shape"))
        .def("__len__", [](const Shape& self) { return self.rank(); })
        .def("__getitem__", [](const Shape& self, std::int64_t index) { return self[index]; })
        .def(
            "__iter__",
            [](const Shape& self) {
                return py::iter(py::cast(ttnn::SmallVector<uint32_t>(self.cbegin(), self.cend())));
            })
        .def(pybind11::self == pybind11::self)
        .def(
            "__repr__",
            [](const Shape& self) {
                std::stringstream ss;
                ss << self;
                return ss.str();
            })
        .def_property_readonly("rank", [](const Shape& self) -> std::size_t { return self.rank(); })
        .def("to_rank", [](const Shape& self, std::size_t new_rank) {
            SmallVector<uint32_t> new_shape(new_rank, 1);

            int cur_idx = static_cast<int>(self.rank()) - 1;
            int new_idx = static_cast<int>(new_rank) - 1;
            for (; cur_idx >= 0 && new_idx >= 0; cur_idx--, new_idx--) {
                new_shape[new_idx] = self[cur_idx];
            }
            for (; cur_idx >= 0; cur_idx--) {
                TT_FATAL(self[cur_idx] == 1, "Can't convert shape rank");
            }

            return ttnn::Shape(std::move(new_shape));
        });
    py::implicitly_convertible<ttnn::SmallVector<uint32_t>, ttnn::Shape>();
}

}  // namespace ttnn::types
