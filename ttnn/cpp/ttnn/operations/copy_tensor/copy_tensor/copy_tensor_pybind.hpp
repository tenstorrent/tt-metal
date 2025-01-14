// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::copy_tensor::detail {

void bind_copy_tensor(pybind11::module& module);

}  // namespace ttnn::operations::copy_tensor::detail
