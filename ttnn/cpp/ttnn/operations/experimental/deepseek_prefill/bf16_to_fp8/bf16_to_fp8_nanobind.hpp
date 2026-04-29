// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8::detail {
namespace nb = nanobind;
void bind_bf16_to_fp8(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_bf16_to_fp8(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
