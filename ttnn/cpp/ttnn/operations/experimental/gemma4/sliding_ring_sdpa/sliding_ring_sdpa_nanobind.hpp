// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::experimental::gemma4 {

namespace nb = nanobind;
void bind_sliding_ring_sdpa(nb::module_& mod);

}  // namespace ttnn::experimental::gemma4
