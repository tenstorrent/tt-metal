// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::untilize_and_split_heads::detail {

void bind_untilize_and_split_heads(nb::module_& mod);

}  // namespace ttnn::operations::experimental::untilize_and_split_heads::detail
