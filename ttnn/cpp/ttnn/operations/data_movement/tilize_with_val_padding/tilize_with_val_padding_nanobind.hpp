// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::data_movement::detail {

namespace nb = nanobind;
void bind_tilize_with_val_padding(nb::module_& mod);
void bind_tilize_with_zero_padding(nb::module_& mod);
}  // namespace ttnn::operations::data_movement::detail
