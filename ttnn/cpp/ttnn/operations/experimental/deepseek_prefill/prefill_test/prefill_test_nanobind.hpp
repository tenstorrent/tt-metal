// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::prefill_test::detail {
namespace nb = nanobind;
void bind_prefill_test(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::prefill_test::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_prefill_test(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
