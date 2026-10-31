// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::detail {
void bind_flash_mqa_decode(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::detail
