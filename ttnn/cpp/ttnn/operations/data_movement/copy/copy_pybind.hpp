// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::data_movement::detail {

void py_bind_copy(pybind11::module& m);
void py_bind_assign(pybind11::module& m);

}  // namespace ttnn::operations::data_movement::detail
