// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::gdn_fused::detail {

namespace nb = nanobind;
void bind_experimental_gdn_fused_operations(nb::module_& mod);

}  // namespace ttnn::operations::experimental::gdn_fused::detail
