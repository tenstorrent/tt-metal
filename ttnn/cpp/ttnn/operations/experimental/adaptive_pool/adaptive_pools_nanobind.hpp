// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::adaptive_pool {
namespace nb = nanobind;

void bind_adaptive_avg_pool2d_operation(nb::module_& mod);
void bind_adaptive_max_pool2d_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::adaptive_pool
