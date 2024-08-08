// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t i = 0;
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto weight_addr = get_arg_val<uint32_t>(i++);
    const auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto start_id = get_arg_val<uint32_t>(i++);
    const auto H = get_arg_val<uint32_t>(i++);
    const auto W = get_arg_val<uint32_t>(i++);
    const auto embedding_dim = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input = tt::CB::c_in0;
    constexpr uint32_t cb_weight = tt::CB::c_in1;
    constexpr uint32_t cb_output = tt::CB::c_out0;

    const uint32_t input_tile_bytes = get_tile_size(cb_input);
    auto input_element_size = input_tile_bytes / 1024;
    auto input_noc_read_size = FACE_WIDTH * input_element_size;

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    const InterleavedAddrGen<input_is_dram> addrg_input = {
        .bank_base_address = input_addr,
        .page_size = input_tile_bytes,
    };

    const uint32_t weight_tile_bytes = get_tile_size(cb_weight);
    auto weight_element_size = weight_tile_bytes / 1024;
    auto weight_noc_read_size = FACE_WIDTH * weight_element_size;

    constexpr bool weight_is_dram = get_compile_time_arg_val(1) == 1;
    const InterleavedAddrGen<weight_is_dram> addrg_weight = {
        .bank_base_address = weight_addr,
        .page_size = weight_tile_bytes,
    };

    uint32_t end_id = start_id + num_tiles_per_core;

    uint32_t Ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t Wt = (W + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t Et = (embedding_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    // for 2d case
    // input: (H, W): int32
    // weight: (num_embeddings, embedding_dim)
    // output: (H, W, embedding_dim)

    for (uint32_t i = start_id; i < end_id; ++i) {
        // output (..., H, W, Embedding_dim)
        // Ht = div_up(H, TILE_HEIGHT)
        // Wt = div_up(W, TILE_HEIGHT)
        // Et = div_up(Embedding_dim, TILE_WIDTH)
        uint32_t et = i % Et;
        uint32_t wt = (i / Et) % Wt;
        uint32_t h = (i / (Wt * Et)) % H;
        uint32_t input_outer_idx = i / (Wt * Et) / H;

        // read input
        uint32_t ht = h / TILE_HEIGHT;

        cb_reserve_back(cb_input, 1);

        uint32_t l1_write_addr = get_write_ptr(cb_input);

        // read input to l1 [0~31]
        uint32_t noc_offset = 0;
        for (uint32_t idx = 0; idx < 2; ++idx) {
            uint32_t input_noc_id = input_outer_idx * Ht * Wt + ht * Wt + wt;
            auto src_noc_addr = get_noc_addr(input_noc_id, addrg_input, noc_offset);

            uint32_t src_offset = get_tilized_idx(h, idx * FACE_WIDTH) * input_element_size;

            noc_async_read(src_noc_addr + src_offset, l1_write_addr, input_noc_read_size);
            noc_async_read_barrier();

            l1_write_addr += input_noc_read_size;
        }

        cb_push_back(cb_input, 1);

        cb_wait_front(cb_input, 1);
        cb_reserve_back(cb_output, 1);

        auto input_ptr = get_read_ptr<int32_t>(cb_input);
        auto output_ptr = get_write_ptr<uint16_t>(cb_output);

        uint32_t w_start = wt * TILE_HEIGHT;
        uint32_t w_end = min(wt * TILE_HEIGHT + TILE_HEIGHT, W);
        for (uint32_t w = w_start; w < w_end; w++) {
            // read idx
            int32_t weight_h = input_ptr[w % TILE_HEIGHT];
            uint32_t weight_ht = weight_h / TILE_HEIGHT;

            // read weight (num_embeddings, embedding_dim)
            uint32_t weight_noc_offset = 0;
            for (uint32_t idx = 0; idx < 2; ++idx) {
                uint32_t weight_noc_id = weight_ht * Et + et;
                auto weight_noc_addr = get_noc_addr(weight_noc_id, addrg_weight, weight_noc_offset);

                uint32_t src_offset = get_tilized_idx(weight_h, idx * FACE_WIDTH) * weight_element_size;
                uint32_t l1_offset = get_tilized_idx(w, idx * FACE_WIDTH) * weight_element_size;

                uint32_t l1_addr = get_write_ptr(cb_output);

                noc_async_read(weight_noc_addr + src_offset, l1_addr + l1_offset, FACE_WIDTH * weight_element_size);
                noc_async_read_barrier();
            }
        }

        cb_pop_front(cb_input, 1);
        cb_push_back(cb_output, 1);
    }
}
