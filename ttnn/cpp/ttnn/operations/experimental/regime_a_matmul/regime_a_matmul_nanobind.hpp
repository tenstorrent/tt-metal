// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::regime_a_matmul::detail {
namespace nb = nanobind;
void bind_regime_a_matmul(nb::module_& mod);

}  // namespace ttnn::operations::experimental::regime_a_matmul::detail
