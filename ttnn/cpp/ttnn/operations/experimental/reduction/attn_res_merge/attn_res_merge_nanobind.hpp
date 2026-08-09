// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::reduction::detail {

namespace nb = nanobind;

void bind_attn_res_merge(nb::module_& mod);
}  // namespace ttnn::operations::experimental::reduction::detail
