// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::topk_xl::detail {

void bind_topk_xl(nb::module_& mod);

}  // namespace ttnn::operations::experimental::topk_xl::detail
