// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::transformer {

void py_bind_rotate_half(pybind11::module& module);

}  // namespace ttnn::operations::experimental::transformer
