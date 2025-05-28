// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

#include "padded_slice.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::padded_slice {
namespace py = pybind11;

void bind_padded_slice(py::module& module);
}  // namespace ttnn::operations::experimental::padded_slice
