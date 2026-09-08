// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::width_to_height_shard::detail {
namespace nb = nanobind;
void bind_width_to_height_shard(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::width_to_height_shard::detail

namespace ttnn::operations::experimental::deepseek::detail {
void bind_width_to_height_shard(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
