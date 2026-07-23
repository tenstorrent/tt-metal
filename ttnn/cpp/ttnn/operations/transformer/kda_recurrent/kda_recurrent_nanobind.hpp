// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::transformer {
void bind_kda_recurrent_step(nanobind::module_& mod);
}  // namespace ttnn::operations::transformer
