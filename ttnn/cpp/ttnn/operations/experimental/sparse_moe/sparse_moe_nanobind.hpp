// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::experimental::sparse_moe::detail {
void bind_sparse_moe_expert(nb::module_& mod);
}
