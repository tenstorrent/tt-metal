// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// batch_fft_writer.cpp — BRISC1 / writer for device-side BATCH FFT.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "batch_fft_common.h"

void kernel_main() {
    const uint32_t out_r_addr      = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr      = get_arg_val<uint32_t>(1);
    const uint32_t base_tile_idx   = get_arg_val<uint32_t>(2);
    const uint32_t batch_per_core  = get_arg_val<uint32_t>(3);
    // arg 4: out_page_size_override.  0 = use CB-derived tile size (legacy);
    //        nonzero = ttnn output buffer page_size (ROW_MAJOR N*elem_size).
    const uint32_t out_page_size_override = get_arg_val<uint32_t>(4);

    // OUTPUT_BF16: when set, convert fp32 STATE → bf16 in CB_OUT_*_BF16 and
    // write bf16 tiles (2048 B) to the output buffers. Default 0 preserves
    // the legacy fp32 fast path.
    constexpr uint32_t OUTPUT_BF16 = get_compile_time_arg_val(0);
    // SUB_N (only needed for OUTPUT_BF16 conversion loop; harmless when 0).
    constexpr uint32_t SUB_N = get_compile_time_arg_val(1);

    const uint32_t   ts = get_tile_size(CB_STATE_R);

    // Output generators.  See reader for full rationale: we MUST use
    // InterleavedAddrGen<true> (NOT *Fast) so the per-bank stride is
    // computed from page_size (aligned to dram_alignment) instead of the
    // hardcoded tile size.  Otherwise ROW_MAJOR tensors with page_size <
    // tile_size scribble at the wrong bank offset once tile_idx wraps
    // past the number of DRAM banks (12 on WH, 8 on BH).
    //
    // NOTE: InterleavedAddrGen has `const` members → no default ctor and
    // no operator= ; construct directly with the right page_size.
    // Use `if constexpr` so the fp32 build never references CB_OUT_R_BF16.
    uint32_t fallback_ts;
    if constexpr (OUTPUT_BF16) {
        fallback_ts = get_tile_size(CB_OUT_R_BF16);
    } else {
        fallback_ts = ts;
    }
    const uint32_t out_ps = out_page_size_override ? out_page_size_override : fallback_ts;
    const InterleavedAddrGen<true> out_r_gen = {
            .bank_base_address = out_r_addr, .page_size = out_ps};
    const InterleavedAddrGen<true> out_i_gen = {
            .bank_base_address = out_i_addr, .page_size = out_ps};

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        const uint32_t tile_idx = base_tile_idx + k;

        cb_wait_front(CB_SYNC, 1);
        cb_wait_front(CB_STATE_R, 1);
        cb_wait_front(CB_STATE_I, 1);

        if constexpr (OUTPUT_BF16) {
            // Convert fp32 STATE → bf16 in CB_OUT_*_BF16, then DMA bf16 tile.
            cb_reserve_back(CB_OUT_R_BF16, 1);
            cb_reserve_back(CB_OUT_I_BF16, 1);
            const uint32_t out_r_bf16_l1 = get_write_ptr(CB_OUT_R_BF16);
            const uint32_t out_i_bf16_l1 = get_write_ptr(CB_OUT_I_BF16);

            volatile tt_l1_ptr uint32_t* const src_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_STATE_R));
            volatile tt_l1_ptr uint32_t* const src_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_STATE_I));
            volatile tt_l1_ptr uint16_t* const dst_r =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const dst_i =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_i_bf16_l1);
            // Truncation (drop low 16 bits). Round-to-nearest-even costs
            // one add + one mask per element; truncation is fine for the
            // first cut and avoids any RNE corner cases.
            for (uint32_t i = 0; i < SUB_N; ++i) {
                dst_r[i] = static_cast<uint16_t>(src_r[i] >> 16);
                dst_i[i] = static_cast<uint16_t>(src_i[i] >> 16);
            }
            cb_push_back(CB_OUT_R_BF16, 1);
            cb_push_back(CB_OUT_I_BF16, 1);

            noc_async_write_tile(tile_idx, out_r_gen, out_r_bf16_l1);
            noc_async_write_tile(tile_idx, out_i_gen, out_i_bf16_l1);
            noc_async_write_barrier();

            cb_pop_front(CB_OUT_R_BF16, 1);
            cb_pop_front(CB_OUT_I_BF16, 1);
        } else {
            noc_async_write_tile(tile_idx, out_r_gen, get_read_ptr(CB_STATE_R));
            noc_async_write_tile(tile_idx, out_i_gen, get_read_ptr(CB_STATE_I));
            noc_async_write_barrier();
        }

        cb_pop_front(CB_SYNC, 1);
        cb_pop_front(CB_STATE_R, 1);
        cb_pop_front(CB_STATE_I, 1);
    }
}
