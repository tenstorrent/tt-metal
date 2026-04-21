// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce::detail {
namespace nb = nanobind;
void bind_post_combine_reduce(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_post_combine_reduce(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
