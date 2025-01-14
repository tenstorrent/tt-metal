// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::copy_tensor {

void py_module(pybind11::module& module);

}  // namespace ttnn::operations::copy_tensor
