// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::pair_pool {

namespace nb = nanobind;
void bind_pair_pool_operation(nb::module_& mod);

}  // namespace ttnn::operations::pair_pool
