// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefetcher_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "prefetcher/dram_prefetcher_nanobind.hpp"

namespace ttnn::operations::prefetcher {

void py_module(nb::module_& mod) { dram_prefetcher::detail::bind_dram_prefetcher(mod); }

}  // namespace ttnn::operations::prefetcher
