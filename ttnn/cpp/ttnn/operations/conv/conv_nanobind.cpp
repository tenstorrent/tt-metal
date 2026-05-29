// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "conv2d/conv2d_nanobind.hpp"

namespace ttnn::operations::conv {

void py_module(nb::module_& mod) {
    // conv1d, conv_transpose2d, and the public conv2d op + prepare_conv_weights/
    // prepare_conv_bias ops were nuked for the agent-regen baseline. Only the
    // shared Conv2dConfig + PaddingMode bindings remain (bind_conv2d trimmed).
    ttnn::operations::conv::conv2d::bind_conv2d(mod);
}
}  // namespace ttnn::operations::conv
