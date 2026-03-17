// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch::detail {
namespace nb = nanobind;
void bind_dispatch(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_dispatch(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
