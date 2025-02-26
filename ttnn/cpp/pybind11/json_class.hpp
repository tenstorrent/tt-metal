// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/reflection.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T>
auto tt_serializable_class(py::module m, auto name, auto desc) {
    return py::class_<T>(m, name, desc)
        .def("to_json", [](const T& self) -> std::string { return tt::stl::json::to_json(self).dump(); })
        .def(
            "from_json",
            [](const std::string& json_string) -> T {
                return tt::stl::json::from_json<T>(nlohmann::json::parse(json_string));
            })
        .def("__repr__", [](const T& self) { return fmt::format("{}", self); });
}
