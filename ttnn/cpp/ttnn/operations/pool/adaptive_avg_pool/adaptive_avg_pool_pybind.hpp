// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::pool {
void py_bind_adaptive_avg_pool(pybind11::module& module);
}  // namespace ttnn::operations::pool
