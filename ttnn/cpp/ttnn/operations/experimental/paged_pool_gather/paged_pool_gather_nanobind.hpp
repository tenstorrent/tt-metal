// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::paged_pool_gather::detail {

void bind_paged_pool_gather(nb::module_& mod);

}  // namespace ttnn::operations::experimental::paged_pool_gather::detail
