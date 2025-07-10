// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/experimental/ccl/llama_common.hpp"

namespace llama_specific {

CoreRangeSet get_custom_cores(uint32_t num_workers, bool row_wise) {
    CoreRangeSet worker_cores;
    std::vector<CoreRange> desired_core_range = {
        CoreRange({5, 3}, {6, 3}), num_workers == 4 ? CoreRange({2, 8}, {3, 8}) : CoreRange({3, 3}, {3, 3})};
    for (const auto& cr : desired_core_range) {
        auto cores = corerange_to_cores(cr, std::nullopt, row_wise);
        for (const auto& core : cores) {
            worker_cores = worker_cores.merge(CoreRangeSet(CoreRange(core, core)));
            if (worker_cores.num_cores() == num_workers) {
                break;
            }
        }
        if (worker_cores.num_cores() == num_workers) {
            break;
        }
    }
    return worker_cores;
}

std::tuple<CoreRangeSet, std::vector<CoreCoord>> get_custom_worker_core_placement(uint32_t num_links) {
    CoreRangeSet cores = get_custom_cores(num_links);
    return {cores, corerange_to_cores(cores)};
}

}  // namespace llama_specific
