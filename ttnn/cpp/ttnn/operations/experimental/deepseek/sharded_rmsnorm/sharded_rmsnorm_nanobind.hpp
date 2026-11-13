// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm::detail {
namespace nb = nanobind;
void bind_sharded_rmsnorm(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm::detail

namespace ttnn::operations::experimental::deepseek::detail {
void bind_sharded_rmsnorm(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
