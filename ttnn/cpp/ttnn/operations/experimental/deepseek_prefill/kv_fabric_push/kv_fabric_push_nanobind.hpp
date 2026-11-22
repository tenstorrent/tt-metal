// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_kv_fabric_push(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
