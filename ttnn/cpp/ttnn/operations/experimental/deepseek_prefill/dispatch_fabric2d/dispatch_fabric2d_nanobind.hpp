// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d::detail {

void bind_experimental_dispatch_fabric2d_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d::detail
