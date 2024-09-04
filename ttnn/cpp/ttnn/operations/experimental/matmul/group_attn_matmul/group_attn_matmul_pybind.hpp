// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::experimental::matmul::detail {

void bind_group_attn_matmul(pybind11::module& module);

}  // namespace ttnn::operations::experimental::matmul::detail
