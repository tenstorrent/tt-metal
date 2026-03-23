// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::transformer {
namespace nb = nanobind;

void bind_dit_rms_norm_unary_fused(nb::module_& mod);

}  // namespace ttnn::operations::experimental::transformer
