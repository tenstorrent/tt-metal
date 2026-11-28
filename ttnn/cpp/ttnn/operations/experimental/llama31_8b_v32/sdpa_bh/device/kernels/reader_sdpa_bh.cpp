// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(7);
    constexpr uint32_t vDHt = get_compile_time_arg_val(8);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(9);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores = get_compile_time_arg_val(13);

    constexpr auto q_args = TensorAccessorArgs<39>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto chunk_start_idx_args = TensorAccessorArgs<attention_sink_args.next_compile_time_args_offset()>();
    constexpr auto cu_window_args = TensorAccessorArgs<chunk_start_idx_args.next_compile_time_args_offset()>();
    constexpr auto q_offset_args = TensorAccessorArgs<cu_window_args.next_compile_time_args_offset()>();
    constexpr auto block_map_args = TensorAccessorArgs<q_offset_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_arg_offset = block_map_args.next_compile_time_args_offset();
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    constexpr uint32_t k_step = Sk_chunk_t * DHt;
    constexpr uint32_t v_step = Sk_chunk_t * vDHt;

    const auto q_reader = TensorAccessor(q_args, get_arg_val<uint32_t>(0));
    const auto k_reader = TensorAccessor(k_args, get_arg_val<uint32_t>(1));
    const auto v_reader = TensorAccessor(v_args, get_arg_val<uint32_t>(2));
    const uint32_t n_steps = get_arg_val<uint32_t>(3);

    for (uint32_t s = 0; s < n_steps; ++s) {
        const uint32_t q_tile = get_arg_val<uint32_t>(4 + 4 * s);
        const uint32_t n_k = get_arg_val<uint32_t>(5 + 4 * s);
        uint32_t k_tile = get_arg_val<uint32_t>(6 + 4 * s);
        uint32_t v_tile = get_arg_val<uint32_t>(7 + 4 * s);
        read_chunk_with_padding<q_tile_bytes>(
            q_reader, cb_q_in, q_tile, Sq_chunk_t, DHt, Sq_chunk_t, DHt, barrier_threshold);
        for (uint32_t k = 0; k < n_k; ++k) {
            read_chunk_with_padding<k_tile_bytes>(
                k_reader, cb_k_in, k_tile, Sk_chunk_t, DHt, Sk_chunk_t, DHt, barrier_threshold, true);
            read_chunk_with_padding<v_tile_bytes>(
                v_reader, cb_v_in, v_tile, Sk_chunk_t, vDHt, Sk_chunk_t, vDHt, barrier_threshold, false);
            k_tile += k_step;
            v_tile += v_step;
        }
    }
}
