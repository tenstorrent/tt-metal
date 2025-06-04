// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::profiler {

namespace nb = nanobind;
void py_module(nb::module_& mod);

}  // namespace ttnn::profiler
