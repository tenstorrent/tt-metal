// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::reduction::tilize_untilize {
void py_bind_tilize_untilize(nanobind::module_& module);
}  // namespace ttnn::operations::reduction::tilize_untilize
