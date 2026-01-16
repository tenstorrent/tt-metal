// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>

#include <tt-metalium/bfloat16.hpp>

// Type Caster Doc:
// https://nanobind.readthedocs.io/en/latest/porting.html#type-casters
// Type Caster example:
// https://github.com/wjakob/nanobind/blob/master/include/nanobind/stl/string.h

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <>
struct type_caster<::bfloat16> {
    NB_TYPE_CASTER(::bfloat16, const_name("bfloat16"));

    bool from_python(handle src, std::uint8_t, cleanup_list*) {
        if (isinstance<nanobind::float_>(src)) {
            this->value = bfloat16(nanobind::cast<float>(src));
            return true;
        }
        if (isinstance<nanobind::int_>(src)) {
            std::int32_t int_value = nanobind::cast<std::int32_t>(src);
            this->value = ::bfloat16(int_value);
            return true;
        }

        return false;
    }

    static handle from_cpp(const ::bfloat16& src, rv_policy, cleanup_list*) {
        return nanobind::float_(static_cast<float>(src)).release();
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
