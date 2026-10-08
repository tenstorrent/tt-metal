// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_ag::detail {

void bind_moe_ag(nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_ag::detail
