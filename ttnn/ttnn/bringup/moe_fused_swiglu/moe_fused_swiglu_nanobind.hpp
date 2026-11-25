// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::bringup::moe_fused_swiglu::detail {
void bind_moe_fused_swiglu(nb::module_& mod);
}

namespace ttnn::operations::bringup::detail {
void bind_moe_fused_swiglu(nb::module_& mod);
}
