// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "flags.hpp"
#include "rect.hpp"

#include "weights.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t k_tiles = CT_ARG("k_tiles_per_b");
    constexpr uint32_t b_tiles = CT_ARG("b_tiles_per_core");
    constexpr uint32_t k_groups = CT_ARG("k_groups");
    constexpr uint32_t chips = CT_ARG("chips");
    constexpr uint32_t lag = CT_ARG("owner_lag");
    constexpr uint32_t reduce_lag = CT_ARG("reduce_lag");
    constexpr uint32_t rows = CT_ARG("rows");
    constexpr uint32_t a_per_group = CT_ARG("a_per_k_group");
    constexpr uint32_t depth = CT_ARG("kred_depth");
    const uint32_t chip = RT_ARG("chip");
    const uint32_t core = RT_ARG("index");
    const uint32_t k_group = core % k_groups;
    const uint32_t n_group = core / k_groups;
    const uint32_t first_column = core * b_tiles;
    CircularBuffer a(CT_ARG("cb_a"));
    CircularBuffer h_segment(CT_ARG("cb_h"));
    Semaphore<> a_free[2] = {Semaphore<>(CT_ARG("sem_a_free")), Semaphore<>(CT_ARG("sem_a_free") + 1)};
    Semaphore<> a_valid[2] = {Semaphore<>(CT_ARG("sem_a_valid")), Semaphore<>(CT_ARG("sem_a_valid") + 1)};
    FlagBoard flags{arena_base(CT_ARG("cb_flags")), 12, 0};
    flags.clear();

    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto w_down =
        TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_down")>(), RT_ARG("w_down_addr"), CT_ARG("tile_bytes_bfp4"));
    load_resident_weights(
        noc,
        w_down,
        CT_ARG("cb_wd"),
        CT_ARG("tile_bytes_bfp4"),
        k_tiles,
        CT_ARG("h_tiles"),
        k_group * k_tiles,
        n_group * CT_ARG("n_tiles_per_b"),
        CT_ARG("n_tiles_per_b"));

    auto peer_x = [&](uint32_t group) { return CT_ARG("b_grid_x_start") + group; };
    const uint32_t peer_y = CT_ARG("b_grid_y_start") + n_group;
    uint32_t credit_posted = 0;
    auto post_credit = [&](uint32_t value) {
        if (value > credit_posted) {
            credit_posted = value;
            for (uint32_t group = 0; group < k_groups; group++) {
                if (group != k_group) {
                    flags.post(noc, peer_x(group), peer_y, FlagBoard::credit + peer_index(k_group, group), value);
                }
            }
        }
    };
    post_credit(depth);

    for (uint32_t step = 0; step < rows + reduce_lag + lag; step++) {
        if (step < rows) {
            const uint32_t slot = step & 1;
            if (step % chips == chip) {
                h_segment.reserve_back(b_tiles);
                for (uint32_t t = 0; t < b_tiles; t++) {
                    noc.async_read(
                        h,
                        CoreLocalMem<uint32_t>(h_segment.get_write_ptr() + t * tile_bytes),
                        tile_bytes,
                        {.page_id = step * CT_ARG("h_tiles") + first_column + t},
                        {});
                }
                noc.async_read_barrier();
                h_segment.push_back(b_tiles);
            }
            a.reserve_back(k_tiles);
            for (uint32_t i = 0; i < a_per_group; i++) {
                const uint32_t a_core = k_group * a_per_group + i;
                a_free[slot].up(
                    noc,
                    CT_ARG("a_grid_x_start") + a_core / CT_ARG("a_rows"),
                    CT_ARG("a_grid_y_start") + a_core % CT_ARG("a_rows"),
                    1);
            }
            a_valid[slot].wait_min(a_per_group * (step / 2 + 1));

            a.push_back(k_tiles);

            for (uint32_t source = 0; source < k_groups - 1; source++) {
                CircularBuffer landing(CT_ARG("cb_kred") + source);
                landing.reserve_back(2 * b_tiles);
                post_credit(step + 2);
                flags.wait_min(FlagBoard::ready + source, step + 1);
                landing.push_back(b_tiles);
            }
        }
        if (chips > 1 && step >= lag && (step - lag) % chips == chip) {
            const uint32_t q = (step - lag) / chips;
            for (uint32_t source = 0; source < chips - 1; source++) {
                CircularBuffer landing(CT_ARG("cb_rs") + source);
                landing.reserve_back(b_tiles);
                Semaphore<>(CT_ARG("sem_rs_ready") + source).wait_min(q + 1);
                landing.push_back(b_tiles);
            }
        }
    }
}
