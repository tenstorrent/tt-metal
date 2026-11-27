// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::compiled_program {
namespace nb = nanobind;
// CompiledProgram, KernelConfigBuffer and ttnn.experimental.create_kernel_config_buffer. Each op that can be compiled
// binds its own compile and run functions (for generic_op, generic_op_nanobind.cpp).
void py_module(nb::module_& mod);
}  // namespace ttnn::compiled_program
