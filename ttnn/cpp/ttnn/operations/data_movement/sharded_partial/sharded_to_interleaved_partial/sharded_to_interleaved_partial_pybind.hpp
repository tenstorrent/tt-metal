// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::data_movement {

void py_bind_sharded_to_interleaved_partial(pybind11::module& module);


}  // namespace ttnn::operations::data_movement::detail
