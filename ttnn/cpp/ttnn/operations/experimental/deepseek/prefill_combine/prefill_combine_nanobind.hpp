// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_combine::detail {
namespace nb = nanobind;
void bind_prefill_combine(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine::detail

namespace ttnn::operations::experimental::deepseek::detail {
void bind_prefill_combine(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
