// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void py_bind_all_reduce_create_qkv_heads(pybind11::module& module);

}  // namespace ttnn::operations::experimental::transformer::detail
