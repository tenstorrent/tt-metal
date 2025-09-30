// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::ternary_backward {

namespace nb = nanobind;
void py_module(nb::module_& mod);
}  // namespace ttnn::operations::ternary_backward
