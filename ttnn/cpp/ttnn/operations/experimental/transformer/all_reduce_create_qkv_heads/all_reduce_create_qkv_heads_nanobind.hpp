// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::transformer::detail {
namespace nb = nanobind;
void bind_all_reduce_create_qkv_heads(nb::module_& mod);
}  // namespace ttnn::operations::experimental::transformer::detail
