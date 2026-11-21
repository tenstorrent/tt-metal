// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::data_movement::detail {

void bind_get_rows_operation(nb::module_& mod);

}  // namespace ttnn::operations::data_movement::detail
