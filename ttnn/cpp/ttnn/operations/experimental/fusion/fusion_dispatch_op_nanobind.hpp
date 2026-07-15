// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::fusion::detail {
void bind_fusion_dispatch_op(nb::module_& mod);
}  // namespace ttnn::operations::experimental::fusion::detail
