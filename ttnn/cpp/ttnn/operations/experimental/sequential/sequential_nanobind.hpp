// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::sequential::detail {

void bind_sequential_operation(nanobind::module_& module);

}  // namespace ttnn::operations::experimental::sequential::detail
