// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::transformer {

void py_bind_sdpa_windowed(pybind11::module& module);

}  // namespace ttnn::operations::transformer
