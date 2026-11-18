// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/detail/nb_dict.h>
#include <nanobind/stl/string.h>

#include <tt-metalium/workload/utils/arg_value.hpp>
#include <tt-metalium/workload/utils/table.hpp>
#include <tt_stl/strong_type.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// The Metal 2.0 name types (KernelSpecName, DFBSpecName, ...) are string StrongTypes; Python passes and reads
// them as str.
template <typename Tag>
struct type_caster<ttsl::StrongType<std::string, Tag>> {
    using NameType = ttsl::StrongType<std::string, Tag>;
    NB_TYPE_CASTER(NameType, const_name("str"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        make_caster<std::string> caster;
        if (!caster.from_python(src, flags, cleanup)) {
            return false;
        }
        value = NameType(std::move(caster.value));
        return true;
    }

    static handle from_cpp(const NameType& name, rv_policy policy, cleanup_list* cleanup) noexcept {
        return make_caster<std::string>::from_cpp(*name, policy, cleanup);
    }
};

// Kernel argument values are uint32_t on the device; Python passes and reads them as int.
template <>
struct type_caster<tt::tt_metal::ArgValue> {
    NB_TYPE_CASTER(tt::tt_metal::ArgValue, const_name("int"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        make_caster<uint32_t> caster;
        if (!caster.from_python(src, flags, cleanup)) {
            return false;
        }
        value = caster.value;
        return true;
    }

    static handle from_cpp(const tt::tt_metal::ArgValue& arg, rv_policy policy, cleanup_list* cleanup) noexcept {
        return make_caster<uint32_t>::from_cpp(arg.to_uint32(), policy, cleanup);
    }
};

// Table<K, V> is a map with unique keys; Python passes and reads it as a dict.
template <typename K, typename V>
struct type_caster<tt::tt_metal::Table<K, V>> : dict_caster<tt::tt_metal::Table<K, V>, K, V> {
    using Base = dict_caster<tt::tt_metal::Table<K, V>, K, V>;

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        if (!Base::from_python(src, flags, cleanup)) {
            return false;
        }
        // Table keeps the first of two Python keys that convert to one C++ key, such as (0, 0) and
        // ttnn.CoreCoord(0, 0), and drops the other value. Reject such a dict instead.
        const Py_ssize_t size = PyObject_Length(src.ptr());
        if (size < 0) {
            PyErr_Clear();
            return false;
        }
        return static_cast<Py_ssize_t>(this->value.size()) == size;
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
