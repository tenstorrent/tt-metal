// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b {

namespace nb = nanobind;
void bind_groupnorm_sc_N_1_HW_C_b_operation(nb::module_& mod);

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b
