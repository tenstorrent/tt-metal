// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail {
namespace nb = nanobind;
void bind_routed_expert_ffn(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_routed_expert_ffn(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
