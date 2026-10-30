// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::kda::select_final_carry::detail {

namespace nb = nanobind;
void bind_select_final_carry(nb::module_&);

}  // namespace ttnn::operations::experimental::kda::select_final_carry::detail
