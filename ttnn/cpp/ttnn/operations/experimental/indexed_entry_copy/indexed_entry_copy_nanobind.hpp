// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::indexed_entry_copy::detail {

namespace nb = nanobind;

void bind_experimental_indexed_entry_copy_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::indexed_entry_copy::detail
