// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::bringup::moe_block_main::fused_head {

namespace nb = nanobind;
// Binds the op into ttnn.bringup.moe_block_main (mod is that submodule); called by ../moe_block_nanobind.cpp.
void bind_fused_head(nb::module_& mod);

}  // namespace ttnn::operations::bringup::moe_block_main::fused_head
