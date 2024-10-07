// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "magic_enum.hpp"

namespace py = pybind11;

template <typename E, typename... Extra>
py::enum_<E> export_enum(const py::handle &scope, std::string name = "", Extra&&... extra) {
    py::enum_<E> enum_type(scope, name.empty() ? magic_enum::enum_type_name<E>().data() : name.c_str(), std::forward<Extra>(extra)...);
    for (const auto &[value, name] : magic_enum::enum_entries<E>()) {
        enum_type.value(name.data(), value);
    }

    return enum_type;
}
