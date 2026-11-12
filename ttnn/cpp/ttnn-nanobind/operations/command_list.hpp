// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::command_list {

namespace nb = nanobind;
void py_module_types(nb::module_& mod);
}  // namespace ttnn::operations::command_list
