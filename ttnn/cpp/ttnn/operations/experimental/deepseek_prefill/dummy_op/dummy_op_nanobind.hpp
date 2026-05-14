// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dummy_op::detail {
namespace nb = nanobind;
void bind_dummy_op(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::dummy_op::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_dummy_op(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
