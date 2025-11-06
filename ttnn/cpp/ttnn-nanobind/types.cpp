// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <tt_stl/small_vector.hpp>
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

#include <umd/device/types/core_coordinates.hpp>

namespace ttnn::types {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::CoreGrid>(mod, "CoreGrid");
    nb::class_<ttnn::Shape>(mod, "Shape");
    nb::class_<ttnn::QueueId>(mod, "QueueId")
        .def(nb::init<uint8_t>())
        .def("__int__", [](const ttnn::QueueId& self) { return static_cast<int>(*self); })
        .def(
            "__repr__",
            [](const ttnn::QueueId& self) { return "QueueId(" + std::to_string(static_cast<int>(*self)) + ")"; })
        .def(nb::self == nb::self,
             nb::sig("def __eq__(self, arg: object, /) -> bool")) // see Typing in nb docs for explanation
        .def("__init__",
            [](ttnn::QueueId* t, nb::int_ arg) {
                // QueueId uses a strong alias so we have to do this manually
                new (t) ttnn::QueueId(static_cast<uint8_t>(arg));
            })
        .def("__init__",
            [](ttnn::QueueId* t, unsigned char arg) {
                // ttnn::maximum's binding wasn't working so this had to be added
                new (t) ttnn::QueueId(static_cast<uint8_t>(arg));
            })
        ;

    export_enum<ttnn::BcastOpMath>(mod, "BcastOpMath");
    export_enum<ttnn::BcastOpDim>(mod, "BcastOpDim");
    auto ct_enum = export_enum<tt::CoreType>(mod, "CoreType");

    // manually do the int cast instead of trying to allow implicit conversion
    ct_enum.def("__int__", [](const tt::CoreType& self) { return nb::int_{static_cast<int>(self)}; });

    nb::implicitly_convertible<nb::int_, ttnn::QueueId>();
    nb::implicitly_convertible<unsigned char, ttnn::QueueId>();

    mod.attr("DRAM_MEMORY_CONFIG") = nb::cast(DRAM_MEMORY_CONFIG);
    mod.attr("L1_MEMORY_CONFIG") = nb::cast(L1_MEMORY_CONFIG);
    mod.attr("DEVICE_STORAGE_TYPE") = nb::cast(DEVICE_STORAGE_TYPE);
    mod.attr("HOST_STORAGE_TYPE") = nb::cast(HOST_STORAGE_TYPE);
}

void py_module(nb::module_& mod) {
    auto py_core_coord = static_cast<nb::class_<ttnn::CoreGrid>>(mod.attr("CoreGrid"));
    py_core_coord
        .def(nb::init<std::size_t, std::size_t>(),
             nb::kw_only(),
             nb::arg("x"), nb::arg("y"))
        .def_prop_ro("x", [](const ttnn::CoreGrid& self) { return self.x; })
        .def_prop_ro("y", [](const ttnn::CoreGrid& self) { return self.y; })
        .def_prop_ro("num_cores", [](const ttnn::CoreGrid& self) { return self.x * self.y; })
        .def("__repr__", [](const ttnn::CoreGrid& self) -> std::string {
            std::stringstream ss;
            ss << self;
            return ss.str();
        });

    auto PyShape = static_cast<nb::class_<ttnn::Shape>>(mod.attr("Shape"));
    PyShape.def(nb::init<const ttnn::SmallVector<uint32_t>&>(), nb::arg("shape"))
        .def("__len__", [](const Shape& self) { return self.rank(); })
        .def("__getitem__", [](const Shape& self, std::int64_t index) { return self[index]; })
        .def(
            "__iter__",  // TODO_NANOBIND: make sure there doesn't need to be an additional cast to SmallVector
            [](const Shape& self) {
                return nb::make_iterator(nb::type<ttnn::Shape>(), "iterator", self.cbegin(), self.cend());
            },
            nb::keep_alive<0, 1>())
        .def(
            nb::self == nb::self,
            nb::sig("def __eq__(self, arg: object, /) -> bool"))  // see Typing in nb docs for explanation
        .def(
            "__repr__",
            [](const Shape& self) {
                std::stringstream ss;
                ss << self;
                return ss.str();
            })
        .def_prop_ro("rank", [](const Shape& self) -> std::size_t { return self.rank(); })
        .def(
            "to_rank",
            [](const Shape& self, std::size_t new_rank) {
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
            })
        .def(nb::init_implicit<ttnn::SmallVector<uint32_t>>());

    nb::implicitly_convertible<ttnn::SmallVector<uint32_t>, ttnn::Shape>();
}

}  // namespace ttnn::types
