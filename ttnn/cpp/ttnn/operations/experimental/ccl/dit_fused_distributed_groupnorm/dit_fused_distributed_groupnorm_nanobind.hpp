// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::ccl {

void bind_dit_fused_distributed_groupnorm(nb::module_& mod);

}  // namespace ttnn::operations::experimental::ccl
