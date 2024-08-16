// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/types.hpp"

#include "conv2d.hpp"

namespace ttnn::operations::conv::conv2d {
    void py_bind_conv2d(pybind11::module& module);
}
