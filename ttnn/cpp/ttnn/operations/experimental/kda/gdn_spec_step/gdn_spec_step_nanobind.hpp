// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::kda::gdn_spec_step::detail {
namespace nb = nanobind;
void bind_gdn_spec_step(nb::module_& mod);
}  // namespace ttnn::operations::experimental::kda::gdn_spec_step::detail
