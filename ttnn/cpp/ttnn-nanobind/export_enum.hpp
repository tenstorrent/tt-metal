// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>

#include <nanobind/nanobind.h>

#include <magic_enum/magic_enum.hpp>

template <typename E, typename... Extra>
nanobind::enum_<E> export_enum(nanobind::handle& scope, std::string name = "", Extra&&... extra) {
    nanobind::enum_<E> enum_type(
        scope, name.empty() ? magic_enum::enum_type_name<E>().data() : name.c_str(), std::forward<Extra>(extra)...);

    for (const auto& [value, name_] : magic_enum::enum_entries<E>()) {
        enum_type.value(name_.data(), value);
    }

    return enum_type;
}
