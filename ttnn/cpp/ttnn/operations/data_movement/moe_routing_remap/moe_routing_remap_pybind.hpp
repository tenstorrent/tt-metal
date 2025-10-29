// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::data_movement::detail {

void py_bind_moe_routing_remap(pybind11::module& module);

}  // namespace ttnn::operations::data_movement::detail
