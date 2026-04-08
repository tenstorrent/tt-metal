// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::conv::conv1d {
namespace nb = nanobind;
void bind_conv1d(nb::module_& mod);
}  // namespace ttnn::operations::conv::conv1d
