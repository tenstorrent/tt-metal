// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_fused_rms_1_1_32_8192(pybind11::module& module);

}  // namespace ttnn::operations::experimental::ccl
