// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::gelu_backward::detail {

namespace nb = nanobind;
void bind_experimental_gelu_backward_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::gelu_backward::detail
