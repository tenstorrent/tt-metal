// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "type_caster.hpp"

namespace py = pybind11;

namespace pybind11 {

namespace detail {

bool type_caster<bfloat16>::load(py::handle src, bool) {
    PyObject *source = src.ptr();
    PyObject *tmp = PyNumber_Float(source);
    if (not tmp) {
        return false;
    }
    double pydouble = PyFloat_AsDouble(tmp);
    value = bfloat16((float)pydouble);
    Py_DECREF(tmp);
    return not PyErr_Occurred();
}

py::handle type_caster<bfloat16>::cast(
    bfloat16 src,
    return_value_policy /* policy */,
    py::handle /* parent */) {
    return PyFloat_FromDouble((double)src.to_float());
}

}  // namespace detail

}  // namespace py
