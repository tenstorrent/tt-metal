// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::ccl {
namespace nb = nanobind;
void bind_strided_reduce_scatter_async(nb::module_& mod);
}  // namespace ttnn::operations::experimental::ccl
