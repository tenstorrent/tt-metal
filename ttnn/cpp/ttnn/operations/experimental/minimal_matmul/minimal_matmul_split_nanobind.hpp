// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::minimal_matmul_split::detail {
namespace nb = nanobind;
void bind_minimal_matmul_split(nb::module_& mod);

}  // namespace ttnn::operations::experimental::minimal_matmul_split::detail
