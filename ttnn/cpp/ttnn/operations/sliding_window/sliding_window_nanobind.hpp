// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::sliding_window {

namespace nb = nanobind;
void bind_sliding_window(nb::module_& mod);
}  // namespace ttnn::operations::sliding_window
