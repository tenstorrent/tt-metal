// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace nb = nanobind;
void bind_routed_combine(nb::module_& mod);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
