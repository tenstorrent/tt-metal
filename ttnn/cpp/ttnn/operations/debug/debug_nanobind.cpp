// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/debug/debug_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/debug/apply_device_delay_nanobind.hpp"

namespace ttnn::operations::debug {

void py_module(nb::module_& mod) { bind_apply_device_delay(mod); }

}  // namespace ttnn::operations::debug
