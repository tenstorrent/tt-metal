// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/bfloat16.hpp"

namespace py = pybind11;

namespace pybind11 {

namespace detail {

// For details see: https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html

// C++ <-> Python conversion for bfloat16
template <>
struct type_caster<bfloat16> {
   public:
    // This macro establishes the name 'bfloat16' in function signatures
    //  and declares a local variable 'value' of type inty
    PYBIND11_TYPE_CASTER(bfloat16, const_name("bfloat16"));

    // Python to C++
    // Converts PyObject to bfloat or return false upon failure
    bool load(py::handle src, bool);

    // C++ to Python
    static py::handle cast(
        bfloat16 src,
        return_value_policy /* policy */,
        py::handle /* parent */);
};

}  // namespace detail

}  // namespace py
