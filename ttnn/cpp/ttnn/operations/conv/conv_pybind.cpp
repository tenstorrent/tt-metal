// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_pybind.hpp"

#include <pybind11/pybind11.h>

#include "conv2d/conv2d_pybind.hpp"
#include "conv_transpose2d/conv_transpose2d_pybind.hpp"
#include "conv1d/conv1d_pybind.hpp"

namespace ttnn::operations::conv {

void py_module(pybind11::module& module) {
    ttnn::operations::conv::conv1d::py_bind_conv1d(module);
    ttnn::operations::conv::conv2d::py_bind_conv2d(module);
    ttnn::operations::conv::conv_transpose2d::py_bind_conv_transpose2d(module);
}
}  // namespace ttnn::operations::conv
