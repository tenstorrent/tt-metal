// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::all_gather_for_matmul::detail {
namespace nb = nanobind;
void bind_all_gather_for_matmul(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::all_gather_for_matmul::detail

namespace ttnn::operations::experimental::deepseek::detail {
void bind_all_gather_for_matmul(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
