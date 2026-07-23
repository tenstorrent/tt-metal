// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::transformer {
namespace nb = nanobind;
void bind_gated_delta_prefill_query(nb::module_& mod);
}  // namespace ttnn::operations::experimental::transformer
