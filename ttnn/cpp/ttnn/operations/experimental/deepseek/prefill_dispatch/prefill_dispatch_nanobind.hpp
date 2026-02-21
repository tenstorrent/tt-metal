// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_dispatch::detail {
namespace nb = nanobind;
void bind_prefill_dispatch(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch::detail

namespace ttnn::operations::experimental::deepseek::detail {
void bind_prefill_dispatch(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
