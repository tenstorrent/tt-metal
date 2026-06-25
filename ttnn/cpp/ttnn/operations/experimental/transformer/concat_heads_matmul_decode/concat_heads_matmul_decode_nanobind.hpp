// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::transformer {

void bind_concat_heads_matmul_decode(nb::module_& mod);

}  // namespace ttnn::operations::experimental::transformer
