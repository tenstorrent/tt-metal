// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {

namespace nb = nanobind;
void bind_reduce_scatter_async(nb::module_& mod);

}  // namespace ttnn::operations::experimental::ccl
