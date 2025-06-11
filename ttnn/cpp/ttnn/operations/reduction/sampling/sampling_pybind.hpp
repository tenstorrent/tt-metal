// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::reduction::detail {
void bind_reduction_sampling_operation(pybind11::module& module);
}  // namespace ttnn::operations::reduction::detail
