// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::variance_w_rm {

namespace nb = nanobind;
void bind_variance_w_rm_operation(nb::module_& mod);

}  // namespace ttnn::operations::variance_w_rm
