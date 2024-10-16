// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include <tuple>
uint32_t nearest_n(uint32_t x, uint32_t n) {
    return ((x + n - 1) / n) * n;
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_runtime_args(int cur_pos,
                                                                    int cur_batch,
                                                                    int core_num,
                                                                    int num_cores_per_batch,
                                                                    uint32_t k_chunk_size) {
    uint32_t valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
    uint32_t pst_value = valid_seq_len / tt::constants::TILE_HEIGHT;
    uint32_t num_chunks_value = valid_seq_len / k_chunk_size;

    uint32_t k_chunk_start = 0;
    uint32_t k_chunk_end = 0;

    if (num_cores_per_batch > int(num_chunks_value)) {
        int chunks_per_core = 1;
        if (core_num >= int(num_chunks_value))
            chunks_per_core = 0;
        k_chunk_start = (num_chunks_value - core_num - 1) * chunks_per_core;
        k_chunk_end = (num_chunks_value - core_num) * chunks_per_core;
    } else {
        int chunks_per_core = num_chunks_value / num_cores_per_batch;
        k_chunk_start = (num_cores_per_batch - core_num - 1) * chunks_per_core;
        k_chunk_end = (num_cores_per_batch - core_num) * chunks_per_core;
        if (core_num == 0) {
            k_chunk_end += (num_chunks_value % num_cores_per_batch);
        }
    }
    return {pst_value, num_chunks_value, k_chunk_start, k_chunk_end};
}
