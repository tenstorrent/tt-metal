// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

namespace nb = nanobind;
void bind_concat_heads_decode(nb::module_& llama31_8b_v3);

}  // namespace ttnn::experimental::llama31_8b_v3
