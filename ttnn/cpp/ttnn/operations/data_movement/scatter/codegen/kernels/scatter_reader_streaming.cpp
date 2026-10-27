// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scatter reader – streaming mode (NCRISC).
// Chosen by the host's select_program_factory() when the full output row does
// not fit L1, or when the tile-row count would leave most of the grid idle.
//
// Work is split by Wt_output across cores (each core handles a subset of output
// columns). For each assigned output column, ALL Wt_index src tiles are scanned
// for matching indices. This mirrors gather streaming (split by output columns).
//
// Protocol (dual-RISC, mirrors interleaved path):
//   Writer (BRISC): loads input→cb_input, writes cb_output→DRAM
//   Reader (NCRISC): waits cb_input, copies to cb_output, scatter, pushes cb_output

#include "scatter_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include <cstdint>

void kernel_main() {
    // Runtime args
    // Buffer addresses are common (per-kernel) runtime args: a program-cache hit then rebinds two
    // slots per kernel instead of two per core.
    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t src_addr = get_common_arg_val<uint32_t>(1);
    // Per-core runtime args
    const uint32_t core_loop_count = get_arg_val<uint32_t>(0);
    const uint32_t tile_width = get_arg_val<uint32_t>(1);
    const uint32_t tile_height = get_arg_val<uint32_t>(2);
    const uint32_t core_id = get_arg_val<uint32_t>(3);
    // Args 4..20 are the fixed-width logical page-map descriptor.

    // Compile-time args
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t cb_src = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_output = get_compile_time_arg_val(5);
    // Index and src share this geometry: validate requires src to carry index's shape.
    constexpr uint32_t Wt_index = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);
    constexpr uint32_t idx_valid_h_last = get_compile_time_arg_val(8);
    constexpr uint32_t idx_valid_w_last = get_compile_time_arg_val(9);
    constexpr uint32_t Ht_per_batch_index = get_compile_time_arg_val(10);
    constexpr uint32_t output_logical_w = get_compile_time_arg_val(11);
    constexpr auto index_ta_args = TensorAccessorArgs<12>();
    constexpr auto src_ta_args = TensorAccessorArgs<index_ta_args.next_compile_time_args_offset()>();
    // This reader overwrites; every reduce the gate admits is served by scatter_reader_bf16_reduce_rm.cpp.
    constexpr uint32_t reduction_mode = get_named_compile_time_arg_val("reduction_mode");
    static_assert(reduction_mode == 0, "this reader does not reduce");

    constexpr uint32_t one_tile = 1;
    const uint32_t tile_width_mask = tile_width - 1;

    // Tensor accessors
    constexpr uint32_t output_tile_bytes = get_tile_size(cb_output);

    constexpr uint32_t index_tile_bytes = get_tile_size(cb_index);
    const auto index_accessor = TensorAccessor(index_ta_args, index_addr, index_tile_bytes);

    constexpr uint32_t src_tile_bytes = get_tile_size(cb_src);
    const auto src_accessor = TensorAccessor(src_ta_args, src_addr, src_tile_bytes);

    constexpr uint32_t output_df_size = output_tile_bytes / get_tile_hw(cb_output);
    constexpr uint32_t index_df_size = index_tile_bytes / get_tile_hw(cb_index);
    constexpr uint32_t src_df_size = src_tile_bytes / get_tile_hw(cb_src);

    constexpr uint32_t face_size = 16;
    constexpr uint32_t FACE_SIZE_MASK = face_size - 1;
    constexpr uint32_t tile_faces = 2;

    Noc noc;
    CircularBuffer in_cb(cb_input);
    CircularBuffer out_cb(cb_output);
    CircularBuffer index_cb(cb_index);
    CircularBuffer src_cb(cb_src);
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    // Work split by Wt_output: each core handles a strided subset of output columns.
    // The output-column tile id restarts at core_id for EACH tile-row h — this core
    // owns the same strided columns {core_id, core_id+num_cores, ...} in every row,
    // and both the `tile_idx == current_output_tile_id` scatter match and the
    // writer's DRAM tile id (h*Wt_output + column) depend on it.
    for (uint32_t h = 0; h < Ht; h++) {
        uint32_t current_output_tile_id = core_id;
        uint32_t h_src = 0;
        const bool has_src_data = map_scatter_input_page<4>(h, h_src);
        const bool is_last_h_tile = has_src_data && ((h_src % Ht_per_batch_index) == (Ht_per_batch_index - 1));
        const uint32_t valid_h = is_last_h_tile ? idx_valid_h_last : tile_height;

        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            // Wait for input tile from writer, copy to output
            in_cb.wait_front(one_tile);
            out_cb.reserve_back(one_tile);

            const uint32_t input_l1 = in_cb.get_read_ptr();
            uint32_t output_l1 = out_cb.get_write_ptr();
            noc.async_read(
                self_ep,
                out_cb,
                output_tile_bytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = input_l1},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            in_cb.pop_front(one_tile);

            if (has_src_data) {
                // Stream through ALL Wt_index src tiles, scatter matching elements
                for (uint32_t ws = 0; ws < Wt_index; ws++) {
                    const bool is_last_w_tile = (ws == Wt_index - 1);
                    const uint32_t valid_w = is_last_w_tile ? idx_valid_w_last : tile_width;

                    // Read index + src tiles (batched, single barrier)
                    index_cb.reserve_back(one_tile);
                    src_cb.reserve_back(one_tile);
                    const uint32_t stile_id = h_src * Wt_index + ws;
                    noc.async_read(
                        index_accessor,
                        index_cb,
                        index_tile_bytes,
                        {.page_id = stile_id, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    noc.async_read(
                        src_accessor,
                        src_cb,
                        src_tile_bytes,
                        {.page_id = stile_id, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    index_cb.push_back(one_tile);
                    src_cb.push_back(one_tile);

                    index_cb.wait_front(one_tile);
                    src_cb.wait_front(one_tile);

                    const uint32_t index_l1 = index_cb.get_read_ptr();
                    const uint32_t src_l1 = src_cb.get_read_ptr();

                    // Scan all elements, scatter those targeting current output tile
                    uint32_t count = 0;
                    for (uint32_t i = 0; i < tile_faces; ++i) {
                        for (uint32_t j = 0; j < tile_faces; ++j) {
                            for (uint32_t k = 0; k < face_size; ++k) {
                                for (uint32_t l = 0; l < face_size; l++) {
                                    const uint32_t row_in_tile = i * face_size + k;
                                    const uint32_t col_in_tile = j * face_size + l;

                                    if (row_in_tile >= valid_h || col_in_tile >= valid_w) {
                                        count++;
                                        continue;
                                    }

                                    const uint32_t global_index = get_value_from_tile(index_l1, count, index_df_size);
                                    // Signed-negative and otherwise invalid
                                    // device index values must not target a
                                    // padded logical column or escape the
                                    // output row's resident L1 allocation.
                                    if (global_index >= output_logical_w) {
                                        count++;
                                        continue;
                                    }
                                    const uint32_t tile_idx = global_index >> __builtin_ctz(tile_width);

                                    if (tile_idx != current_output_tile_id) {
                                        count++;
                                        continue;
                                    }

                                    const uint32_t src_value = get_value_from_tile(src_l1, count, src_df_size);
                                    const uint32_t index_in_local_tile = global_index & tile_width_mask;
                                    const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);
                                    const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;

                                    const uint32_t local_index = which_row * (face_size * face_size) + k * face_size +
                                                                 which_col + i * (tile_width * face_size);

                                    write_value_to_tile(output_l1, local_index, output_df_size, src_value);
                                    count++;
                                }
                            }
                        }
                    }

                    index_cb.pop_front(one_tile);
                    src_cb.pop_front(one_tile);
                }  // Wt_index loop
            }  // has_src_data

            // Push completed output tile
            out_cb.push_back(one_tile);
            current_output_tile_id += num_cores;
        }  // core_loop
    }  // Ht loop
}
