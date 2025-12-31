// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nlohmann/json.hpp>

#include <tt_stl/reflection.hpp>

template <typename T>
auto tt_serializable_class(nanobind::module_& mod, auto name, auto desc) {
    return nanobind::class_<T>(mod, name, desc)
        .def("to_json", [](const T& self) -> std::string { return ttsl::json::to_json(self).dump(); })
        .def(
            "from_json",
            [](const std::string& json_string) -> T {
                return ttsl::json::from_json<T>(nlohmann::json::parse(json_string));
            })
        .def("__repr__", [](const T& self) { return fmt::format("{}", self); });
}
