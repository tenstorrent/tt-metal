// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::normalization::detail {

namespace nb = nanobind;
void bind_normalization_rms_norm(nb::module_& mod);

}  // namespace ttnn::operations::normalization::detail
