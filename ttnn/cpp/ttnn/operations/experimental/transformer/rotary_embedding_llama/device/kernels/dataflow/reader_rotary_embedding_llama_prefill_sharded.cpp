// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    auto batch_start = get_arg(args::batch_start);
    auto batch_end = get_arg(args::batch_end);
    auto seq_t_start = get_arg(args::seq_t_start);
    auto seq_t_end = get_arg(args::seq_t_end);

    constexpr auto n_heads = get_arg(args::n_heads);
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr bool freq_per_head = get_arg(args::freq_per_head) == 1;
    constexpr auto cos_Ht = get_arg(args::cos_Ht);
    constexpr auto sin_Ht = get_arg(args::sin_Ht);
    constexpr auto rotary_Ht = get_arg(args::rotary_Ht);

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(tensor::input);

    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_cos(dfb::cos);
    DataflowBuffer dfb_sin(dfb::sin);
    DataflowBuffer dfb_trans_mat(dfb::trans_mat);

    const uint32_t input_tile_bytes = dfb_input.get_entry_size();

    // ------------------------------------------------------------------
    // Transformation matrix
    // ------------------------------------------------------------------
    // Read transformation matrix in CB (only once, because it will be reused)
    // NOTE: the `#ifdef`/scope pairing below replaces the legacy `if constexpr (trans_mat_use_global_cb)`
    // — promoted to a preprocessor gate so the conditional `tensor::trans_mat` binding parses away when
    // trans_mat is borrowed. The `{ }` scope keeps the body indentation unchanged from the legacy branch.
#ifdef TRANS_MAT_USE_GLOBAL_CB
    {
        // This block helps performance significantly. Although the `else` block works in all cases,
        // removing this `if` block negatively affects performance. In most cases, we should land
        // in this block unless users are improperly sharding `trans_mat`, e.g., replicating it on
        // fewer cores than available in the chosen core grid.
        dfb_trans_mat.reserve_back(onetile);
        dfb_trans_mat.push_back(onetile);
    }
#else
    {
        // Non-height-sharded/fewer-#shards-than-cores-height-sharded cases
        const auto s3 = TensorAccessor(tensor::trans_mat);

        dfb_trans_mat.reserve_back(onetile);
        uint32_t trans_mat_l1_write_addr = dfb_trans_mat.get_write_ptr();
        noc.async_read(
            s3, CoreLocalMem<uint32_t>(trans_mat_l1_write_addr), dfb_trans_mat.get_entry_size(), {.page_id = 0}, {});
        noc.async_read_barrier();
        dfb_trans_mat.push_back(onetile);
    }
#endif

    // ------------------------------------------------------------------
    // Cos / Sin
    // ------------------------------------------------------------------
    // NOTE: the `#ifdef`/scope pairing below replaces the legacy `if constexpr (cos_sin_sharded)` — promoted
    // to a preprocessor gate so the conditional `tensor::cos`/`tensor::sin` bindings parse away on the
    // borrowed fast path. The `{ }` scope keeps the body indentation unchanged from the legacy branch.
#ifdef COS_SIN_SHARDED
    {
        // Sharded cos/sin: fast (1 seq tile/core, L1 view) vs reload (multi seq, read via TensorAccessor)
#if COS_SIN_SHARDED_RELOAD == 1
        const uint32_t cos_tile_bytes = dfb_cos.get_entry_size();
        const uint32_t sin_tile_bytes = dfb_sin.get_entry_size();
        const auto s1 = TensorAccessor(tensor::cos);
        const auto s2 = TensorAccessor(tensor::sin);
#endif

        for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
            for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
                for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
                    dfb_cos.reserve_back(Wt);
                    dfb_sin.reserve_back(Wt);
                    dfb_input.reserve_back(Wt);
#if COS_SIN_SHARDED_RELOAD == 1
                    uint32_t cos_l1_write_addr = dfb_cos.get_write_ptr();
                    uint32_t sin_l1_write_addr = dfb_sin.get_write_ptr();
                    uint32_t cos_curr_idx = freq_per_head ? (head_num * cos_Ht * Wt + seq_tile * Wt) : (seq_tile * Wt);
                    uint32_t sin_curr_idx = freq_per_head ? (head_num * sin_Ht * Wt + seq_tile * Wt) : (seq_tile * Wt);
                    for (uint32_t j = 0; j < Wt; ++j) {
                        noc.async_read(
                            s1,
                            CoreLocalMem<uint32_t>(cos_l1_write_addr),
                            cos_tile_bytes,
                            {.page_id = cos_curr_idx},
                            {});
                        noc.async_read(
                            s2,
                            CoreLocalMem<uint32_t>(sin_l1_write_addr),
                            sin_tile_bytes,
                            {.page_id = sin_curr_idx},
                            {});
                        cos_curr_idx++;
                        sin_curr_idx++;
                        cos_l1_write_addr += cos_tile_bytes;
                        sin_l1_write_addr += sin_tile_bytes;
                    }
#endif
                    uint32_t input_l1_write_addr = dfb_input.get_write_ptr();
                    uint32_t input_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                    for (uint32_t j = 0; j < Wt; ++j) {
                        noc.async_read(
                            s0,
                            CoreLocalMem<uint32_t>(input_l1_write_addr),
                            input_tile_bytes,
                            {.page_id = input_curr_idx},
                            {});
                        input_curr_idx++;
                        input_l1_write_addr += input_tile_bytes;
                    }
                    noc.async_read_barrier();
                    dfb_cos.push_back(Wt);
                    dfb_sin.push_back(Wt);
                    dfb_input.push_back(Wt);
                }
            }
        }
    }
#else
    {
        // Interleaved cos/sin (trans_mat may still be sharded).
        const uint32_t cos_tile_bytes = dfb_cos.get_entry_size();
        const auto s1 = TensorAccessor(tensor::cos);

        const uint32_t sin_tile_bytes = dfb_sin.get_entry_size();
        const auto s2 = TensorAccessor(tensor::sin);

        for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
            uint32_t sin_l1_write_addr = 0;
            uint32_t cos_l1_write_addr = 0;
#if RELOAD_IMPL == 0
            if (my_cos_sin_tiles > 0) {
                dfb_sin.reserve_back(my_cos_sin_tiles);
                dfb_cos.reserve_back(my_cos_sin_tiles);
                sin_l1_write_addr = dfb_sin.get_write_ptr();
                cos_l1_write_addr = dfb_cos.get_write_ptr();
            }
#endif

            // To make sure the sin/cos row are read only once
            uint32_t sin_cos_row_cnt = 0;
            bool done_sin_cos = false;

            for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
                for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
#if RELOAD_IMPL == 1
                    dfb_sin.reserve_back(Wt);
                    dfb_cos.reserve_back(Wt);
                    uint32_t sin_l1_write_addr = dfb_sin.get_write_ptr();
                    uint32_t cos_l1_write_addr = dfb_cos.get_write_ptr();
#endif

                    dfb_input.reserve_back(Wt);
                    uint32_t input_l1_write_addr = dfb_input.get_write_ptr();
                    uint32_t input_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                    uint32_t cos_curr_idx;
                    uint32_t sin_curr_idx;
                    if constexpr (freq_per_head) {
                        cos_curr_idx = head_num * cos_Ht * Wt + seq_tile * Wt;
                        sin_curr_idx = head_num * sin_Ht * Wt + seq_tile * Wt;
                    } else {
                        cos_curr_idx = seq_tile * Wt;
                        sin_curr_idx = seq_tile * Wt;
                    }
                    for (uint32_t j = 0; j < Wt; ++j) {
                        // Read input into CB
                        noc.async_read(
                            s0,
                            CoreLocalMem<uint32_t>(input_l1_write_addr),
                            input_tile_bytes,
                            {.page_id = input_curr_idx},
                            {});
                        input_curr_idx++;
                        input_l1_write_addr += input_tile_bytes;

                        if (!done_sin_cos) {
                            noc.async_read(
                                s2,
                                CoreLocalMem<uint32_t>(sin_l1_write_addr),
                                sin_tile_bytes,
                                {.page_id = sin_curr_idx},
                                {});
                            noc.async_read(
                                s1,
                                CoreLocalMem<uint32_t>(cos_l1_write_addr),
                                cos_tile_bytes,
                                {.page_id = cos_curr_idx},
                                {});
                            sin_curr_idx++;
                            cos_curr_idx++;
                            sin_l1_write_addr += sin_tile_bytes;
                            cos_l1_write_addr += cos_tile_bytes;
                        }
                    }

                    noc.async_read_barrier();
                    dfb_input.push_back(Wt);
#if RELOAD_IMPL == 1
                    dfb_sin.push_back(Wt);
                    dfb_cos.push_back(Wt);
#else

                    if (!done_sin_cos) {
                        dfb_sin.push_back(Wt);
                        dfb_cos.push_back(Wt);

                        // Update sin_cos_row_cnt
                        sin_cos_row_cnt++;

                        if (sin_cos_row_cnt == my_rotary_seq_tiles) {
                            done_sin_cos = true;
                        }
                    }
#endif
                }
            }
        }
    }
#endif
}
