// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::detail {
void bind_engram_gate(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
