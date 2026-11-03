// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj writer: store_wave. Per wave, per tile row m of this core's rows: one untilized block of
// core_n_tiles tiles (32 rows x core_n_tiles * 32 bf16) -> 32 row segments of y, row 32 * m + i at byte offset
// line * seg_bytes. One flush per block before the pop, one barrier at the end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "moe_down_proj_plan.h"

void kernel_main() {
    constexpr uint32_t cb_output_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_counts_writer = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_waves = get_compile_time_arg_val(3);
    constexpr uint32_t num_m_groups = get_compile_time_arg_val(4);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t group = get_arg_val<uint32_t>(1);
    const uint32_t line = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t seg_bytes = core_n_tiles * tile_rows * sizeof(uint16_t);
    const uint32_t seg_offset = line * seg_bytes;
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_bytes);

    cb_wait_front(cb_counts_writer, 1);
    uint32_t counts[MOE_NUM_EXPERTS];
    {
        volatile tt_l1_ptr uint32_t* landed =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_counts_writer));
        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            counts[e] = landed[e];
        }
    }
    const MoePlan plan(counts, tensor_row_tiles, num_waves, num_m_groups);

    for (uint32_t wave = 0; wave < num_waves; ++wave) {
        if (plan.wave_empty(wave)) {
            continue;
        }
        const uint32_t row_lo = plan.group_first(wave, group);
        const uint32_t my_rows = plan.group_rows(wave, group);
        for (uint32_t r = 0; r < my_rows; ++r) {
            cb_wait_front(cb_output_rows, core_n_tiles);
            const uint32_t block_l1 = get_read_ptr(cb_output_rows);
            const uint32_t first_row = (row_lo + r) * tile_rows;
            for (uint32_t i = 0; i < tile_rows; ++i) {
                noc_async_write(
                    block_l1 + i * seg_bytes, out_accessor.get_noc_addr(first_row + i, seg_offset), seg_bytes);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_output_rows, core_n_tiles);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_counts_writer, 1);
}
