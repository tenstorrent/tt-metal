// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::conv::conv2d {
namespace nb = nanobind;
void bind_conv2d(nb::module_& mod);
}  // namespace ttnn::operations::conv::conv2d
