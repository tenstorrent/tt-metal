// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::layer_norm_w_rm {

namespace nb = nanobind;
void bind_layer_norm_w_rm_operation(nb::module_& mod);

}  // namespace ttnn::operations::layer_norm_w_rm