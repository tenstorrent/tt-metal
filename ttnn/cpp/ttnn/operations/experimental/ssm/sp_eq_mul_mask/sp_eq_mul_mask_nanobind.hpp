// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ssm::detail {

namespace nb = nanobind;
void bind_sp_eq_mul_mask(nb::module_& mod);

}  // namespace ttnn::operations::experimental::ssm::detail
