// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::data_movement {

void py_bind_interleaved_to_sharded_partial(pybind11::module& module);

}  // namespace ttnn::operations::data_movement
