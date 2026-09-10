// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::kda::affine_exclusive_scan::detail {

namespace nb = nanobind;
void bind_affine_exclusive_scan(nb::module_&);

}  // namespace ttnn::operations::experimental::kda::affine_exclusive_scan::detail
