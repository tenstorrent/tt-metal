// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_nanobind.hpp"

#include <nanobind/nanobind.h>

namespace ttnn::operations::conv {

void py_module(nb::module_& mod) {
    // TODO(nuked-op conv2d): conv2d/conv1d/conv_transpose2d bindings removed for eval.
    // Restore the bind_* calls here once the conv ops are recreated.
    (void)mod;
}
}  // namespace ttnn::operations::conv
