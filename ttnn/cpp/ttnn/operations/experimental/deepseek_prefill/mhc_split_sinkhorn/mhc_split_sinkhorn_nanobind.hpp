// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::deepseek_prefill::mhc_split_sinkhorn::detail {
void bind_experimental_mhc_split_sinkhorn_operation(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_split_sinkhorn::detail
