// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::reduction::detail {

namespace nb = nanobind;
void bind_std_var_reductions(nb::module_& mod);
}  // namespace ttnn::operations::reduction::detail
