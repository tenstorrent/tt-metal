// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::detail {
namespace nb = nanobind;
void bind_deepseek_moe_gate(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {
void bind_deepseek_moe_gate(::nanobind::module_& mod);
}
