// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail {
namespace nb = nanobind;
void bind_fused_experts(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {
void bind_fused_experts(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::moe::detail
