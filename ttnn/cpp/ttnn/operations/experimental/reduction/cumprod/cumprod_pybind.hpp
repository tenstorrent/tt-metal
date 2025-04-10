// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/reduction/cumprod/cumprod.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::experimental::reduction::cumprod::detail {

void bind_cumprod_operation(py::module& module);

}  // namespace ttnn::operations::experimental::reduction::cumprod::detail
