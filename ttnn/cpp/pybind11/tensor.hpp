// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::tensor {

    void pytensor_module_types(pybind11::module & m_tensor);
    void pytensor_module(pybind11::module & m_tensor);
    void py_module_types(py::module& module);
    void py_module(py::module& module);

} // namespace ttnn::tensor
