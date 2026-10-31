// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

namespace nb = nanobind;
// Binds the op into ttnn.experimental.moe_block (mod is that submodule); called by ../moe_block_nanobind.cpp.
void bind_sparse_dispatch_2d(nb::module_& mod);

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
