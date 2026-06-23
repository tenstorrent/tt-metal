// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::examples {
namespace nb = nanobind;
void bind_bh_dram_read_operation(nb::module_& mod);
}  // namespace ttnn::operations::examples
