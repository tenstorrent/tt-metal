// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "conv2d/conv2d_nanobind.hpp"
#include "conv_transpose2d/conv_transpose2d_nanobind.hpp"
#include "conv1d/conv1d_nanobind.hpp"

namespace ttnn::operations::conv {

void py_module(nb::module_& mod) {
    ttnn::operations::conv::conv1d::bind_conv1d(mod);
    ttnn::operations::conv::conv2d::bind_conv2d(mod);
    ttnn::operations::conv::conv_transpose2d::bind_conv_transpose2d(mod);
}
}  // namespace ttnn::operations::conv
