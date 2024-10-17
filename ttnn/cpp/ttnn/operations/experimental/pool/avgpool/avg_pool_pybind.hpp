// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace py = pybind11;
namespace ttnn::operations::experimental::pool {
namespace detail {

void bind_avg_pool2d(py::module& module);

}  // namespace detail
}  // namespace ttnn::operations::experimental::pool
