// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::fr_detail {

namespace nb = nanobind;

void bind_fused_rotate(nb::module_& mod);

}  // namespace ttnn::operations::experimental::fr_detail
