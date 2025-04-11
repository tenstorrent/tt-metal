// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <enchantum/enchantum.hpp>

template <typename E, typename... Extra>
nanobind::enum_<E> export_enum(const nanobind::handle& scope, const std::string& name = "", Extra&&... extra) {
    nanobind::enum_<E> enum_type(
        scope, name.empty() ? enchantum::type_name<E>.data() : name.c_str(), std::forward<Extra>(extra)...);
    for (const auto [value, name_] : enchantum::entries_generator<E>) {
        enum_type.value(name_.data(), value);
    }

    return enum_type;
}
