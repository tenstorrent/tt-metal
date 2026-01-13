// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::transformer {
namespace nb = nanobind;
void bind_dit_layernorm_post_all_gather(nb::module_& mod);
}  // namespace ttnn::operations::experimental::transformer
