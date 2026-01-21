// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::tensor_accessor_args {
namespace nb = nanobind;
void py_module_types(nb::module_& mod);
void py_module(nb::module_& mod);

}  // namespace ttnn::tensor_accessor_args
