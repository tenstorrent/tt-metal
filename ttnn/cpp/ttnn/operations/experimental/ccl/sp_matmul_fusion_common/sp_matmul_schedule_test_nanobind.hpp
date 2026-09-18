// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {
namespace nb = nanobind;
// ttnn.experimental.sp_matmul_schedule_test and ttnn.experimental.sp_matmul_program_config (test-only helpers for
// the sequence-parallel matmul fusion).
void bind_sp_matmul_schedule_test(nb::module_& mod);
}  // namespace ttnn::operations::experimental::ccl
