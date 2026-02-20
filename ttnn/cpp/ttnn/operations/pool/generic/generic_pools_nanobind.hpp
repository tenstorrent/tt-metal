// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::pool {

namespace nb = nanobind;

void bind_max_pool2d_operation(nb::module_& mod);
void bind_avg_pool2d_operation(nb::module_& mod);
void py_module(nb::module_& mod);

}  // namespace ttnn::operations::pool
