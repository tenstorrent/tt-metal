// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::normalization::detail {

void bind_all_gather_rms_norm(nb::module_& mod);

}  // namespace ttnn::operations::normalization::detail
