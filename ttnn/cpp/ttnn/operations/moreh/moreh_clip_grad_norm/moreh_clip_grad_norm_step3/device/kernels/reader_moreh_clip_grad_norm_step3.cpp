// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto clip_coef_clamped_addr = get_arg_val<uint32_t>(i++);
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_clip_coef_clamped = cb_id++;

    constexpr uint32_t onetile = 1;

    // input
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input_addrg = TensorAccessor(input_args, input_addr);

    // clip_coef_clamped
    constexpr auto coef_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto coef_addrg = TensorAccessor(coef_args, clip_coef_clamped_addr);

    Noc noc;
    DataflowBuffer dfb_input(cb_id_input);
    DataflowBuffer dfb_clip_coef(cb_id_clip_coef_clamped);
    const auto input_tile_bytes = get_tile_size(cb_id_input);
    const auto clip_coef_tile_bytes = get_tile_size(cb_id_clip_coef_clamped);

    // clip_coef_clamped
    dfb_clip_coef.reserve_back(onetile);
    noc.async_read(coef_addrg, dfb_clip_coef, clip_coef_tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb_clip_coef.push_back(onetile);

    // input
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        dfb_input.reserve_back(onetile);
        noc.async_read(input_addrg, dfb_input, input_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_input.push_back(onetile);
    }

}  // void kernel_main()
