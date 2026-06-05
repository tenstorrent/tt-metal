// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::conv1d_depthwise::detail {
void bind_conv1d_depthwise(nb::module_& mod);
}  // namespace ttnn::operations::experimental::conv1d_depthwise::detail
