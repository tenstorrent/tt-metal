// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::detail {

void bind_unified_routed_expert_ffn(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_unified_routed_expert_ffn(::nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
