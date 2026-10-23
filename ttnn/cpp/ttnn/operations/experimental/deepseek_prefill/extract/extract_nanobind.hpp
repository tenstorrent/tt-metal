// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::extract::detail {
namespace nb = nanobind;
void bind_extract(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::extract::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_extract(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
