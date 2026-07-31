// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::all_gather_regime_a_matmul_async::detail {
namespace nb = nanobind;
void bind_all_gather_regime_a_matmul_async(nb::module_& mod);

}  // namespace ttnn::operations::experimental::all_gather_regime_a_matmul_async::detail
