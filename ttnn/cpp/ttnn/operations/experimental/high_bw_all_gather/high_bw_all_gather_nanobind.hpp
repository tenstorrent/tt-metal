// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather::detail {

namespace nb = nanobind;
void bind_experimental_high_bw_all_gather_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::high_bw_all_gather::detail
