// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::dram_prefetcher::detail {

namespace nb = nanobind;
void bind_dram_prefetcher(nb::module_& mod);

}  // namespace ttnn::operations::dram_prefetcher::detail
