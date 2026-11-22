// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::bringup::moe_block_main {

namespace nb = nanobind;
// Creates the ttnn.bringup.moe_block_main submodule of ttnn.bringup (ttnn._ttnn.operations.bringup.moe_block_main) and
// binds common/ and every op into it.
void bind_moe_block(nb::module_& bringup);

}  // namespace ttnn::operations::bringup::moe_block_main
