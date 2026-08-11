// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::small_m_matmul::detail {
namespace nb = nanobind;
void bind_small_m_matmul(nb::module_& mod);

}  // namespace ttnn::operations::experimental::small_m_matmul::detail
