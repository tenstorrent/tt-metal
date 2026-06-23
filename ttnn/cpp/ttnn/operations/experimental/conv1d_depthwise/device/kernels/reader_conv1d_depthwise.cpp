// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Fill one fp32 tile (1024 elements) in a CB with a constant. Layout-agnostic: every
// element gets the same value, so face order is irrelevant.
FORCE_INLINE void fill_tile_fp32_at(uint32_t l1_addr, float value) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr float*>(l1_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = value;
    }
}

// Fill the K scalar tap tiles once into scalar_cb (resident, depth K). The taps are constant
// for the whole op, so filling them per-block (the old path) burned ~44% of reader time.
FORCE_INLINE void fill_scalar_tiles(uint32_t scalar_cb_id, uint32_t K, uint32_t tile_bytes) {
    cb_reserve_back(scalar_cb_id, K);
    const uint32_t base = get_write_ptr(scalar_cb_id);
    for (uint32_t j = 0; j < K; ++j) {
        const uint32_t tap_bits = get_common_arg_val<uint32_t>(j);
        float tap;
        __builtin_memcpy(&tap, &tap_bits, sizeof(float));
        fill_tile_fp32_at(base + j * tile_bytes, tap);
    }
    cb_push_back(scalar_cb_id, K);
}

FORCE_INLINE void zero_region_u32(uint32_t l1_addr, uint32_t num_bytes) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    const uint32_t words = num_bytes >> 2;
    for (uint32_t i = 0; i < words; ++i) {
        ptr[i] = 0;
    }
}

void kernel_main() {
    constexpr uint32_t act_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);      // real channels
    constexpr uint32_t C_pad = get_compile_time_arg_val(3);  // padded to TILE_WIDTH multiple
    constexpr uint32_t stride = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t T_pad = get_compile_time_arg_val(6);
    constexpr uint32_t T_out = get_compile_time_arg_val(7);
    constexpr uint32_t block_h_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(9);
    constexpr uint32_t B = get_compile_time_arg_val(10);

    constexpr auto src_args = TensorAccessorArgs<11>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);  // global flattened output row (b*T_out + t_out)
    const uint32_t num_rows = get_arg_val<uint32_t>(2);   // valid output rows for this core

    const auto src = TensorAccessor(src_args, src_addr, C * 4);

    constexpr uint32_t BLOCK_T = block_h_tiles * 32;
    constexpr uint32_t block_w_tiles = C_pad / 32;
    constexpr uint32_t block_num_tiles = block_h_tiles * block_w_tiles;
    constexpr uint32_t stick_bytes = C * 4;
    constexpr uint32_t padded_stick_bytes = C_pad * 4;

    const uint32_t num_blocks = (num_rows + BLOCK_T - 1) / BLOCK_T;
    constexpr uint32_t scalar_tile_bytes = 1024 * 4;  // fp32 32x32 tile

    // Taps are constant for the whole op — fill the K scalar tiles once (resident, depth K).
    // Compute reads them by tap index without popping.
    fill_scalar_tiles(scalar_cb_id, K, scalar_tile_bytes);

    // The op is read-bound: compute sits ~92% idle waiting on the per-tap activation windows.
    // The K tap windows overlap heavily (input pages base*stride + i*stride + j), so the per-tap
    // reader re-reads each element ~K times from interleaved DRAM. When B==1 the block's input
    // pages form one contiguous run [base*stride .. base*stride + (BLOCK_T-1)*stride + K-1]: read
    // that union ONCE from DRAM, then gather each tap window from L1 scratch. Cuts DRAM ~K×.
    // (B>1 breaks the single-run assumption — keep the per-tap DRAM path.)
    constexpr bool coalesce = (B == 1);

    if constexpr (coalesce) {
        constexpr uint32_t union_sticks = (BLOCK_T - 1) * stride + K;
        const uint32_t scratch_ptr = get_write_ptr(scratch_cb_id);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t base_in = (row_start + blk * BLOCK_T) * stride;
            // Read the union once. Pages past T_pad (tail of the last partial block) stay zero;
            // their output rows are beyond num_rows and the writer discards them.
            zero_region_u32(scratch_ptr, union_sticks * padded_stick_bytes);
            for (uint32_t u = 0; u < union_sticks; ++u) {
                const uint32_t in_page = base_in + u;
                if (in_page >= T_pad) {
                    break;
                }
                noc_async_read(src.get_noc_addr(in_page), scratch_ptr + u * padded_stick_bytes, stick_bytes);
            }
            noc_async_read_barrier();

            for (uint32_t j = 0; j < K; ++j) {
                cb_reserve_back(act_cb_id, block_num_tiles);
                const uint32_t wptr = get_write_ptr(act_cb_id);
                if constexpr (stride == 1) {
                    // Tap window = union sticks [j .. j+BLOCK_T-1] — one contiguous L1->L1 copy.
                    noc_async_read(
                        get_noc_addr(scratch_ptr + j * padded_stick_bytes), wptr, BLOCK_T * padded_stick_bytes);
                } else {
                    // Strided gather: output row i reads union stick (i*stride + j), from L1.
                    for (uint32_t i = 0; i < BLOCK_T; ++i) {
                        noc_async_read(
                            get_noc_addr(scratch_ptr + (i * stride + j) * padded_stick_bytes),
                            wptr + i * padded_stick_bytes,
                            padded_stick_bytes);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(act_cb_id, block_num_tiles);
            }
        }
        return;
    }

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base_row = row_start + blk * BLOCK_T;
        for (uint32_t j = 0; j < K; ++j) {
            // Activation window for tap j: BLOCK_T sticks gathered from absolute input pages.
            cb_reserve_back(act_cb_id, block_num_tiles);
            const uint32_t wptr = get_write_ptr(act_cb_id);
            zero_region_u32(wptr, BLOCK_T * padded_stick_bytes);
            for (uint32_t i = 0; i < BLOCK_T; ++i) {
                const uint32_t local = blk * BLOCK_T + i;
                if (local >= num_rows) {
                    break;  // remaining rows stay zero-padded
                }
                const uint32_t g = base_row + i;
                const uint32_t b = g / T_out;
                const uint32_t t_out = g % T_out;
                const uint32_t in_page = b * T_pad + t_out * stride + j;
                noc_async_read(src.get_noc_addr(in_page), wptr + i * padded_stick_bytes, stick_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(act_cb_id, block_num_tiles);
        }
    }
}
