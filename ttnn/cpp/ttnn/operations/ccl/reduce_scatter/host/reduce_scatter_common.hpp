// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace ttnn {
namespace ccl {
namespace reduce_scatter_detail {

struct WorkerTransferInfo {
    WorkerTransferInfo(
        std::vector<uint32_t> pages_per_full_chunk_per_worker, uint32_t num_links, uint32_t num_workers);

    uint32_t get_num_pages_per_full_chunk(uint32_t link, uint32_t worker_idx) const;

    std::vector<uint32_t> pages_per_full_chunk_per_worker;
    uint32_t num_links;
    uint32_t num_workers;
};

} // namespace reduce_scatter_detail
} // namespace ccl
} // namespace ttnn
