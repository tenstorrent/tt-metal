// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::reduction::detail {
namespace nb = nanobind;
void bind_deepseek_moe_fast_reduce_nc(nb::module_& mod);
}  // namespace ttnn::operations::experimental::reduction::detail
