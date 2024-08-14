// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"


namespace ttnn::operations::experimental::transformer::detail {

void bind_create_qkv_heads(pybind11::module& module);
} // namespace ttnn::operations::experimental::transformer::detail
