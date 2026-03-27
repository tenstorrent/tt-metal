// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_moe_post_combine_reduce::detail {
namespace nb = nanobind;
void bind_deepseek_moe_post_combine_reduce(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_moe_post_combine_reduce::detail
