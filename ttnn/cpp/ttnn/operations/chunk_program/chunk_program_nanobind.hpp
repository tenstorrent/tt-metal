// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::chunk_program {

namespace nb = nanobind;
void bind_chunk_program_operation(nb::module_& mod);

}  // namespace ttnn::operations::chunk_program
