// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

#include "weights.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t a_tiles = CT_ARG("a_tiles_per_core");
    constexpr uint32_t a_row_bytes = CT_ARG("k_tiles_per_b") * CT_ARG("tile_bytes_act");
    constexpr uint32_t b_cores = CT_ARG("b_rows");
    const uint32_t k_group = RT_ARG("index") / CT_ARG("a_per_k_group");
    const uint32_t b_x = CT_ARG("b_grid_x_start") + k_group;
    const Rect b_column{b_x, CT_ARG("b_grid_y_start"), b_x, CT_ARG("b_grid_y_end"), CT_ARG("b_rows")};
    const uint32_t b_slots = arena_base(CT_ARG("cb_n"));
    const uint32_t a_offset = (RT_ARG("index") % CT_ARG("a_per_k_group")) * a_tiles * CT_ARG("tile_bytes_act");
    CircularBuffer a_out(CT_ARG("cb_a_out"));
    Semaphore<> a_free[2] = {Semaphore<>(CT_ARG("sem_a_free")), Semaphore<>(CT_ARG("sem_a_free") + 1)};
    Semaphore<> a_valid[2] = {Semaphore<>(CT_ARG("sem_a_valid")), Semaphore<>(CT_ARG("sem_a_valid") + 1)};

    const auto w_gate_up = TensorAccessor(
        TensorAccessorArgs<CT_ARG("ta_w_gate_up")>(), RT_ARG("w_gate_up_addr"), CT_ARG("tile_bytes_bfp4"));
    load_resident_weights(
        noc,
        w_gate_up,
        CT_ARG("cb_wg"),
        CT_ARG("tile_bytes_bfp4"),
        CT_ARG("h_tiles"),
        CT_ARG("gate_up_tiles"),
        0,
        2 * a_tiles * RT_ARG("index"),
        2 * a_tiles);

    for (uint32_t m = 0; m < CT_ARG("rows"); m++) {
        const uint32_t slot = m & 1;
        a_out.wait_front(a_tiles);

        a_free[slot].wait_min(b_cores * (m / 2 + 1));

        b_column.write_each(
            noc,
            a_out.get_read_ptr(),
            b_slots + slot * a_row_bytes + a_offset,
            a_tiles * CT_ARG("tile_bytes_act"),
            a_valid[slot]);
        noc.async_write_barrier();

        a_out.pop_front(a_tiles);
    }
}
