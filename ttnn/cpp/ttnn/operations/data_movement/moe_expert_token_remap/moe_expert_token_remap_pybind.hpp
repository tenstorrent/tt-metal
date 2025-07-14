// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::data_movement::detail {

void py_bind_moe_expert_token_remap(pybind11::module& module);

}  // namespace ttnn::operations::data_movement::detail
