// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::per_core_allocation {
void py_module(nanobind::module_& m);
}  // namespace ttnn::per_core_allocation
