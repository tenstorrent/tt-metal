// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::data_movement {

namespace nb = nanobind;
void bind_interleaved_to_sharded_partial(nb::module_& mod);

}  // namespace ttnn::operations::data_movement
