// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::detail {

void bind_deepseek_moe_post_combine_reduce(nb::module_& mod);

}  // namespace ttnn::operations::experimental::detail