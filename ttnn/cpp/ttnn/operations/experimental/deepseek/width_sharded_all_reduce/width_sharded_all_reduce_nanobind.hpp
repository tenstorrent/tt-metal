// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::width_sharded_all_reduce::detail {
namespace nb = nanobind;
void bind_width_sharded_all_reduce(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::width_sharded_all_reduce::detail

namespace ttnn::operations::experimental::deepseek::detail {
void bind_width_sharded_all_reduce(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
