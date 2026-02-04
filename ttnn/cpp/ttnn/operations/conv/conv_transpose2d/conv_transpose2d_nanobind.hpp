// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::conv::conv_transpose2d {
namespace nb = nanobind;
void bind_conv_transpose2d(nb::module_& mod);
}  // namespace ttnn::operations::conv::conv_transpose2d
