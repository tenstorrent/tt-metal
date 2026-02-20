// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::reduction::accumulation::detail {
namespace nb = nanobind;
void bind_reduction_ema_operation(nb::module_& mod);

}  // namespace ttnn::operations::reduction::accumulation::detail
