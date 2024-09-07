// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::experimental::paged_cache::detail {

namespace py = pybind11;

void bind_experimental_paged_cache_operations(py::module& module);
}  // namespace ttnn::operations::experimental::paged_cache::detail
