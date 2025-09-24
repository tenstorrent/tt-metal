// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pytypes.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tt-metalium/bfloat16.hpp>

namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
struct type_caster<bfloat16> {
    PYBIND11_TYPE_CASTER(bfloat16, _("bfloat16"));

    bool load(handle src, bool) {
        if (isinstance<pybind11::float_>(src)) {
            value = bfloat16(src.cast<float>());
            return true;
        } else if (isinstance<pybind11::int_>(src)) {
            int32_t int_value = src.cast<int32_t>();
            value = bfloat16(int_value);
            return true;
        }

        return false;
    }

    static handle cast(const bfloat16& src, return_value_policy, handle) {
        return pybind11::float_(static_cast<float>(src)).release();
    }
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
