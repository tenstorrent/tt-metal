// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::minimal_matmul::detail {

void bind_minimal_matmul_split(nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::minimal_matmul::detail
