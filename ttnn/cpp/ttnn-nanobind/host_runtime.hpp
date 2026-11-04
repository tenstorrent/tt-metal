// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::host_runtime {
namespace nb = nanobind;
void py_module(nb::module_& mod);

}  // namespace ttnn::host_runtime
