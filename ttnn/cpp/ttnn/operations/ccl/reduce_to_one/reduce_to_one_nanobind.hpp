// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::ccl {

void bind_reduce_to_one(nanobind::module_& module);

}  // namespace ttnn::operations::ccl
