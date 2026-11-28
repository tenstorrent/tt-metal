// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::pack_fp8_rows {

namespace nb = nanobind;
void bind_pack_fp8_rows_operation(nb::module_& mod);

}  // namespace ttnn::operations::pack_fp8_rows
