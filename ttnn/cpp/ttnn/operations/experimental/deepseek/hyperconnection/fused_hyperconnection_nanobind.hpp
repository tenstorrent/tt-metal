// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection::detail {
namespace nb = nanobind;
void bind_fused_hyperconnection(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::hyperconnection::detail

namespace ttnn::operations::experimental::deepseek::detail {
void bind_fused_hyperconnection(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
