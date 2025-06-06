// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace tt::operations::primary {

namespace nb = nanobind;
void py_module(nb::module_& mod) {}

} // namespace tt::operations::primary 
