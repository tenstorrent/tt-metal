// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::ternary::detail {

void bind_where(pybind11::module& pymodule);

}  // namespace ttnn::operations::experimental::ternary::detail
