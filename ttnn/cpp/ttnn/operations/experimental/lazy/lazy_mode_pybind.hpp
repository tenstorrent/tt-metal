// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::experimental::lazy {

void bind_lazy_mode(pybind11::module& module);

}  // namespace ttnn::experimental::lazy
