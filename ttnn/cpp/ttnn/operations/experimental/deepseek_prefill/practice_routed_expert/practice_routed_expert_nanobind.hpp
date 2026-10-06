// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert::detail {
namespace nb = nanobind;
void bind_practice_routed_expert(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_practice_routed_expert(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
