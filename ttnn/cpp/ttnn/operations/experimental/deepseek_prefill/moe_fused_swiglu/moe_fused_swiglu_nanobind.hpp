// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::detail {
void bind_moe_fused_swiglu(nb::module_& mod);
}

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_moe_fused_swiglu(nb::module_& mod);
}
