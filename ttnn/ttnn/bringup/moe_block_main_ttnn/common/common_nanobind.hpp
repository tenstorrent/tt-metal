// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::bringup::moe_block_main {

namespace nb = nanobind;
// Config structs, shared host math and MoEBlockResources into ttnn.bringup.moe_block_main.
void bind_common(nb::module_& mod);

}  // namespace ttnn::operations::bringup::moe_block_main
