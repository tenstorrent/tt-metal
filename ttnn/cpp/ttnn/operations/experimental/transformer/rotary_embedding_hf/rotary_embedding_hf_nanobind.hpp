// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::transformer {

void bind_rotary_embedding_hf(nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::transformer
