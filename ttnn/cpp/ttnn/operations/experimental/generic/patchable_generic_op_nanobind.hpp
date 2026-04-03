// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::generic::detail {
void bind_patchable_generic_op(nb::module_& mod);
}  // namespace ttnn::operations::experimental::generic::detail
