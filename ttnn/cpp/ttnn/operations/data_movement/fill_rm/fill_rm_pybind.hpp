// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::data_movement {

void bind_fill_rm(pybind11::module& module);

}  // namespace ttnn::operations::data_movement
