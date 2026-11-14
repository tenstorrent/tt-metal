// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::experimental::llama31_8b {

namespace nb = nanobind;
void bind_llama31_8b(nb::module_& mod);

}  // namespace ttnn::experimental::llama31_8b
