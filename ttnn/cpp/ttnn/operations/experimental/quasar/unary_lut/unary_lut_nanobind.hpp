// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::quasar::detail {

void bind_unary_lut(nb::module_& mod);

}  // namespace ttnn::operations::experimental::quasar::detail
