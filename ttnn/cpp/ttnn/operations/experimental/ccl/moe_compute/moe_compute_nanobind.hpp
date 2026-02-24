// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {
namespace nb = nanobind;
void bind_moe_compute(nb::module_& mod);
void bind_get_moe_combine_cores(nb::module_& mod);
}  // namespace ttnn::operations::experimental::ccl
