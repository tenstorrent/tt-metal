// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::transformer {

void bind_rotary_embedding_fused_qk(nb::module_& mod);

}  // namespace ttnn::operations::experimental::transformer
