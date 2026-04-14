// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::mla::detail {
namespace nb = nanobind;
void bind_matmul_wo(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek::mla::detail
