// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::generic {
namespace nb = nanobind;

// Binds the Metal 2.0 ProgramSpec types that the ProgramSpec overload of ttnn.generic_op takes.
void bind_program_spec_types(nb::module_& mod);

}  // namespace ttnn::operations::generic
