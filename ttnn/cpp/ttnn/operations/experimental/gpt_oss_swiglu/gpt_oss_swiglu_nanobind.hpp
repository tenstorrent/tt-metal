// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::gpt_oss_swiglu::detail {
namespace nb = nanobind;
void bind_gpt_oss_swiglu(nb::module_& mod);
}  // namespace ttnn::operations::experimental::gpt_oss_swiglu::detail

namespace ttnn::operations::experimental::detail {
void bind_gpt_oss_swiglu(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::detail
