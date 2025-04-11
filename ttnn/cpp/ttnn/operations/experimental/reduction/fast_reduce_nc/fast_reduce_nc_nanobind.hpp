// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::reduction::detail {

namespace nb = nanobind;

void bind_fast_reduce_nc(nb::module_& mod);
}  // namespace ttnn::operations::experimental::reduction::detail
