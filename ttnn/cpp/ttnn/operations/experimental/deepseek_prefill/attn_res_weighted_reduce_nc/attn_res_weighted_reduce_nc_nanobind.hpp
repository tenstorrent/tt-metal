// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_weighted_reduce_nc::detail {

namespace nb = nanobind;

void bind_attn_res_weighted_reduce_nc(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_weighted_reduce_nc::detail
