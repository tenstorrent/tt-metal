// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::data_movement {

namespace nb = nanobind;
void bind_copy_to_memory_config(nb::module_& mod);

}  // namespace ttnn::operations::data_movement
