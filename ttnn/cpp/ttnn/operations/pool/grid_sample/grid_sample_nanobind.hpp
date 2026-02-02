// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::grid_sample {
namespace nb = nanobind;
void bind_grid_sample(nb::module_& mod);

}  // namespace ttnn::operations::grid_sample
