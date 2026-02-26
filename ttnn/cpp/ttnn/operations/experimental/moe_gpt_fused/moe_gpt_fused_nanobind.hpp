// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::moe_gpt_fused::detail {
namespace nb = nanobind;
void bind_moe_gpt_fused(nb::module_& mod);

}  // namespace ttnn::operations::experimental::moe_gpt_fused::detail
