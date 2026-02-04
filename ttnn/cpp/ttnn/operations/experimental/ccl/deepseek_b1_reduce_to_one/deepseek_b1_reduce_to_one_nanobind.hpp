// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::ccl {

void bind_deepseek_b1_reduce_to_one(nanobind::module_& module);

}  // namespace ttnn::operations::experimental::ccl
