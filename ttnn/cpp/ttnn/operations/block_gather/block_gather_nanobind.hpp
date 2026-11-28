// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::block_gather {

namespace nb = nanobind;
void bind_block_gather_operation(nb::module_& mod);

}  // namespace ttnn::operations::block_gather
