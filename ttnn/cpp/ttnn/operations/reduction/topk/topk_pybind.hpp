// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::reduction::detail {
void bind_reduction_topk_operation(pybind11::module& module);
}  // namespace ttnn::operations::reduction::detail
