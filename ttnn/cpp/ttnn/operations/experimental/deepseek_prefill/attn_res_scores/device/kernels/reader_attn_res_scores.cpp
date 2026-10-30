// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_scores/device/kernels/attn_res_scores_items.hpp"

void kernel_main() {
    constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    // Tensor-parallel partials side by side on the last dim; compute sums them.
    constexpr uint32_t num_partials = get_compile_time_arg_val(2);
    constexpr uint32_t kSplitArgs = 3;
    constexpr auto stats_args = TensorAccessorArgs<kSplitArgs + 5>();

    const auto stats_addr = get_common_arg_val<uint32_t>(0);

    constexpr uint32_t cb_stats = 0;
    constexpr uint32_t tile_bytes = get_tile_size(cb_stats);

    Noc noc;
    CircularBuffer cb_stats_obj(cb_stats);
    auto stats_accessor = TensorAccessor(stats_args, stats_addr);

    // The reader does not need to know which sites an item covers, only its row; the
    // site count passed here is irrelevant to it, so any non-zero value serves.
    for_each_attn_res_scores_item<kSplitArgs, 1>([&](const AttnResScoresItem& item) {
        // Candidate-major, partials innermost: a candidate's partials are adjacent pages.
        cb_stats_obj.reserve_back(num_candidates * num_partials);
        for (uint32_t c = 0; c < num_candidates; ++c) {
            for (uint32_t p = 0; p < num_partials; ++p) {
                noc.async_read(
                    stats_accessor,
                    cb_stats_obj,
                    tile_bytes,
                    {.page_id = (c * Ht + item.row) * num_partials + p},
                    {.offset_bytes = (c * num_partials + p) * tile_bytes});
            }
        }
        noc.async_read_barrier();
        cb_stats_obj.push_back(num_candidates * num_partials);
    });
}
