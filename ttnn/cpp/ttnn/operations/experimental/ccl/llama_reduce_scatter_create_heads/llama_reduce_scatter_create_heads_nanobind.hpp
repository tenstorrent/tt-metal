// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {
namespace nb = nanobind;
void bind_llama_rs_create_heads(nb::module_& mod);
}  // namespace ttnn::operations::experimental::ccl
