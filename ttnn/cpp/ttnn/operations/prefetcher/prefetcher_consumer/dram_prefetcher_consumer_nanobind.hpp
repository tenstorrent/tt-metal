// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::dram_prefetcher_consumer::detail {

namespace nb = nanobind;

void bind_dram_prefetcher_consumer(nb::module_& mod);

}  // namespace ttnn::operations::dram_prefetcher_consumer::detail
