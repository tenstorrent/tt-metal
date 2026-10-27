// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::transformer {

void bind_fused_recurrent_gated_delta_rule(nb::module_& mod);

}  // namespace ttnn::operations::transformer
