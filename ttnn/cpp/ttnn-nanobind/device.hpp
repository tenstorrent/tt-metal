// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::device {
namespace nb = nanobind;

void py_device_module_types(nb::module_& mod);
void py_device_module(nb::module_& mod);

}  // namespace ttnn::device
