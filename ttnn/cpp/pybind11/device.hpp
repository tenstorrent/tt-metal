// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

namespace py = pybind11;

namespace ttnn::device {

    void py_device_module_types(py::module& module);
    void py_device_module(py::module& module);

} // namespace ttnn::device
