// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {
namespace nb = nanobind;
void bind_all_gather_matmul_async(nb::module_& mod);

}  // namespace ttnn::operations::experimental::ccl
