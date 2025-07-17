// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <enchantum/type_name.hpp>

template <typename E, typename... Extra>
pybind11::enum_<E> export_enum(const pybind11::handle& scope, std::string name = "", Extra&&... extra) {
    pybind11::enum_<E> enum_type(
        scope, name.empty() ? enchantum::type_name<E>.data() : name.c_str(), std::forward<Extra>(extra)...);
    for (const auto [value, name] : enchantum::entries_generator<E>) {
        enum_type.value(name.data(), value);
    }

    return enum_type;
}
