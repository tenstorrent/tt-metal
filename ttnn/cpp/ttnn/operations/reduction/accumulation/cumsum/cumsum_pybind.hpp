// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::reduction::accumulation::detail {
namespace py = pybind11;

void bind_reduction_cumsum_operation(py::module& module);

}  // namespace ttnn::operations::reduction::accumulation::detail
