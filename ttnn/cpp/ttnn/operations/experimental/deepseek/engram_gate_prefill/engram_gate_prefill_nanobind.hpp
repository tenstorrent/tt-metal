// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace nanobind {
class module_;
}

namespace ttnn::operations::experimental::deepseek::detail {

void bind_engram_gate_prefill(::nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek::detail
