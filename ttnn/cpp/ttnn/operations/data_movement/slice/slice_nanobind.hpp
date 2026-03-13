// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::data_movement::detail {

namespace nb = nanobind;
void bind_slice(nb::module_& mod);
void bind_slice_descriptor(nb::module_& mod);
}  // namespace ttnn::operations::data_movement::detail
