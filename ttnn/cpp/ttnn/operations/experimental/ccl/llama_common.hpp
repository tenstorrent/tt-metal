#pragma once

namespace llama_specific {

CoreRangeSet get_custom_cores(const uint32_t num_workers, bool row_wise);

std::tuple<CoreRangeSet, std::vector<CoreCoord>> get_custom_worker_core_placement(uint32_t num_links);

}  // namespace llama_specific
