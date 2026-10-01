// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv::detail {
namespace nb = nanobind;
void bind_fused_lightning_select_kv(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv::detail

namespace ttnn::operations::experimental::deepseek::detail {
void bind_fused_lightning_select_kv(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
