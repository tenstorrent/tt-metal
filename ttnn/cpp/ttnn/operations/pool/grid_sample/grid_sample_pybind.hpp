// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::grid_sample {

void py_bind_grid_sample(pybind11::module& module);

namespace detail {
void bind_grid_sample(pybind11::module& module);
void bind_prepare_grid_sample_grid(pybind11::module& module);
}  // namespace detail

}  // namespace ttnn::operations::grid_sample
