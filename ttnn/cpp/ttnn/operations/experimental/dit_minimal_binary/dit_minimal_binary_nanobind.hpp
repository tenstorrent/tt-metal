// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::dit_minimal_binary::detail {
namespace nb = nanobind;
void bind_dit_minimal_binary(nb::module_& mod);
}  // namespace ttnn::operations::experimental::dit_minimal_binary::detail
