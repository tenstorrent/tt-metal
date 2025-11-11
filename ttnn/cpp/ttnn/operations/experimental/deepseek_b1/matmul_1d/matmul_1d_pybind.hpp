// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_b1::matmul_1d::detail {

void bind_matmul_1d(pybind11::module& module);

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d::detail
