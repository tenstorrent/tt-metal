// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::moe_gate_mm::detail {
namespace nb = nanobind;
void bind_moe_gate_mm(nb::module_& mod);

}  // namespace ttnn::operations::experimental::moe_gate_mm::detail
