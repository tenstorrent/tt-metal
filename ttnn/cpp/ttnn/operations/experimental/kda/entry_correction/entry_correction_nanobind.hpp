// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::kda::entry_correction::detail {

namespace nb = nanobind;
void bind_entry_correction(nb::module_&);

}  // namespace ttnn::operations::experimental::kda::entry_correction::detail
