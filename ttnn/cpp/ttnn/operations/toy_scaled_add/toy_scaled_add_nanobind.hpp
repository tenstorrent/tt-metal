// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::toy_scaled_add {

namespace nb = nanobind;
void bind_toy_scaled_add_operation(nb::module_& mod);

}  // namespace ttnn::operations::toy_scaled_add
