// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::transformer {
namespace nb = nanobind;
void bind_dit_layernorm_pre_all_gather(nb::module_& mod);
}  // namespace ttnn::operations::experimental::transformer
