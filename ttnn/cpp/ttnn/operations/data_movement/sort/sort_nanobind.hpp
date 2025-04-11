// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::reduction::sort::detail {

namespace nb = nanobind;
void bind_reduction_sort_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::reduction::sort::detail
