// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::prefetcher {

void py_module(pybind11::module& module);

}  // namespace ttnn::operations::prefetcher
