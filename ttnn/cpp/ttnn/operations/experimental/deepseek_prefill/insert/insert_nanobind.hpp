// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::insert::detail {
namespace nb = nanobind;
void bind_insert(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::insert::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_insert(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
