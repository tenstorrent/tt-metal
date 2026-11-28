// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::block_keys {

namespace nb = nanobind;
void bind_block_keys_operation(nb::module_& mod);

}  // namespace ttnn::operations::block_keys
