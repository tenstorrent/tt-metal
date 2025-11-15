// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::isin::detail {

void bind_isin_operation(py::module& module);

}  // namespace ttnn::operations::experimental::isin::detail
