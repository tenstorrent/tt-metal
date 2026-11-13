// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

namespace nb = nanobind;
// Binds the op into ttnn.experimental.moe_block (mod is that submodule); called by ../moe_block_nanobind.cpp.
void bind_flat_routed_expert(nb::module_& mod);

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
