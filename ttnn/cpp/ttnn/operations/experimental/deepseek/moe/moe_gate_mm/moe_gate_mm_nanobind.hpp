// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm::detail {
namespace nb = nanobind;
void bind_moe_gate_mm(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {
void bind_moe_gate_mm(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::moe::detail
