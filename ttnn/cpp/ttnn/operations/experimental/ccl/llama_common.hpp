// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#pragma once

#include <tt-metalium/core_coord.hpp>

namespace llama_specific {

CoreRangeSet get_custom_cores(uint32_t num_workers, bool row_wise = true);

std::tuple<CoreRangeSet, std::vector<CoreCoord>> get_custom_worker_core_placement(uint32_t num_links);

}  // namespace llama_specific
