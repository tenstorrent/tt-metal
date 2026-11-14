// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

namespace nb = nanobind;
void bind_expert_slot_copy(nb::module_& mod);

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy
