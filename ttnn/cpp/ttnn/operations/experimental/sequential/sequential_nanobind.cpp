// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sequential_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "sequential.hpp"

namespace ttnn::operations::experimental::sequential::detail {

namespace nb = nanobind;

void bind_sequential_operation(nb::module_& /*module*/) {
    // Sequential execution is currently handled entirely in Python.
    // The ttnn.sequential function is defined in ttnn/ttnn/operations/sequential.py
    //
    // Future: Bind C++ fused sequential execution when CB chaining is implemented.
}

}  // namespace ttnn::operations::experimental::sequential::detail
