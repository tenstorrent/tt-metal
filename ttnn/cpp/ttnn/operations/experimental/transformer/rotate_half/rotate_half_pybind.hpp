// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::experimental::transformer {

void py_bind_rotate_half(pybind11::module& module);

}  // namespace ttnn::operations::experimental::transformer
