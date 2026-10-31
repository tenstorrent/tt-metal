// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::moe_block {

namespace nb = nanobind;
// Creates the ttnn.experimental.moe_block submodule of `experimental` and binds common/ and every op into it.
void bind_moe_block(nb::module_& experimental);

}  // namespace ttnn::operations::experimental::moe_block
