// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill::detail {

void bind_fused_experts_prefill(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_fused_experts_prefill(::nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
