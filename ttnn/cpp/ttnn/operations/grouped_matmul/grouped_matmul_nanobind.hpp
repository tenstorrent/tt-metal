// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::grouped_matmul {

namespace nb = nanobind;
void bind_grouped_matmul_operation(nb::module_& mod);

}  // namespace ttnn::operations::grouped_matmul
