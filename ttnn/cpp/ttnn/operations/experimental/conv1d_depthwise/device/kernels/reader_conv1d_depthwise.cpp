// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Compile-time config at file scope so the block-reading helpers can reference it directly.
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

constexpr uint32_t BLOCK_T = block_h_tiles * 32;
constexpr uint32_t block_w_tiles = C_pad / 32;
constexpr uint32_t block_num_tiles = block_h_tiles * block_w_tiles;
constexpr uint32_t stick_bytes = C * 4;
constexpr uint32_t padded_stick_bytes = C_pad * 4;
constexpr uint32_t union_sticks = (BLOCK_T - 1) * stride + K;

// B==1: input pages form one contiguous run, so coalesce the K overlapping tap windows into a
// single DRAM read. B>1 uses the per-tap path.
constexpr bool coalesce = (B == 1);

// Fill one fp32 tile (1024 elements) with a constant value.
FORCE_INLINE void fill_tile_fp32_at(uint32_t l1_addr, float value) {
    CoreLocalMem<float> ptr(l1_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = value;
    }
}

// Fill the K scalar tap tiles once into scalar_cb (resident, depth K); taps are constant for the op.
FORCE_INLINE void fill_scalar_tiles(CircularBuffer& scalar_cb, uint32_t tile_bytes) {
    scalar_cb.reserve_back(K);
    const uint32_t base = scalar_cb.get_write_ptr();
    for (uint32_t j = 0; j < K; ++j) {
        const uint32_t tap_bits = get_common_arg_val<uint32_t>(j);
        float tap;
        __builtin_memcpy(&tap, &tap_bits, sizeof(float));
        fill_tile_fp32_at(base + j * tile_bytes, tap);
    }
    scalar_cb.push_back(K);
}

// Zero num_bytes at the CB write pointer via the NoC zero API. Barriers before returning, so the
// zeros are visible to subsequent reads/writes into the region.
FORCE_INLINE void zero_cb_region(const Noc& noc, const CircularBuffer& cb, uint32_t num_bytes) {
    noc.async_write_zeros(cb, num_bytes);
    noc.write_zeros_l1_barrier();
}

// Coalesced (B==1) path for one block: read the block's union of input pages once from DRAM into
// L1 scratch, then gather each tap window from L1. Pages past T_pad are left untouched (they only
// feed output rows the writer discards). Scratch is zeroed once in kernel_main, not here.
template <typename SrcAccessor>
FORCE_INLINE void read_block_coalesced(
    const Noc& noc, const SrcAccessor& src, CircularBuffer& act_cb, CircularBuffer& scratch_cb, uint32_t base_row) {
    const uint32_t scratch_base = scratch_cb.get_write_ptr();
    const uint32_t base_in = base_row * stride;
    for (uint32_t u = 0; u < union_sticks; ++u) {
        const uint32_t in_page = base_in + u;
        if (in_page >= T_pad) {
            break;
        }
        noc.async_read(src, scratch_cb, stick_bytes, {.page_id = in_page}, {.offset_bytes = u * padded_stick_bytes});
    }
    noc.async_read_barrier();

    for (uint32_t j = 0; j < K; ++j) {
        act_cb.reserve_back(block_num_tiles);
        UnicastEndpoint self;
        if constexpr (stride == 1) {
            // Tap window = union sticks [j .. j+BLOCK_T-1]: one contiguous L1->L1 copy.
            noc.async_read(
                self,
                act_cb,
                BLOCK_T * padded_stick_bytes,
                experimental::local_addr(scratch_base + j * padded_stick_bytes, noc.get_noc_id()),
                {.offset_bytes = 0});
        } else {
            // Strided gather: output row i reads union stick (i*stride + j), from L1.
            for (uint32_t i = 0; i < BLOCK_T; ++i) {
                noc.async_read(
                    self,
                    act_cb,
                    padded_stick_bytes,
                    experimental::local_addr(scratch_base + (i * stride + j) * padded_stick_bytes, noc.get_noc_id()),
                    {.offset_bytes = i * padded_stick_bytes});
            }
        }
        noc.async_read_barrier();
        act_cb.push_back(block_num_tiles);
    }
}

// Per-tap (general / B>1) path for one block: each tap window is gathered directly from absolute
// input pages in DRAM.
template <typename SrcAccessor>
FORCE_INLINE void read_block(
    const Noc& noc,
    const SrcAccessor& src,
    CircularBuffer& act_cb,
    uint32_t blk,
    uint32_t base_row,
    uint32_t num_rows) {
    for (uint32_t j = 0; j < K; ++j) {
        // Activation window for tap j: BLOCK_T sticks gathered from absolute input pages.
        act_cb.reserve_back(block_num_tiles);
        // act_cb is double-buffered, so only the first 2 reservations need their channel-pad tail
        // [C*4, C_pad*4) zeroed; later reservations reuse already-defined slots.
        if (blk * K + j < 2) {
            zero_cb_region(noc, act_cb, BLOCK_T * padded_stick_bytes);
        }
        for (uint32_t i = 0; i < BLOCK_T; ++i) {
            const uint32_t local = blk * BLOCK_T + i;
            if (local >= num_rows) {
                break;  // remaining rows stay zero-padded
            }
            const uint32_t g = base_row + i;
            const uint32_t b = g / T_out;
            const uint32_t t_out = g % T_out;
            const uint32_t in_page = b * T_pad + t_out * stride + j;
            noc.async_read(src, act_cb, stick_bytes, {.page_id = in_page}, {.offset_bytes = i * padded_stick_bytes});
        }
        noc.async_read_barrier();
        act_cb.push_back(block_num_tiles);
    }
}

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<11>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);  // global flattened output row (b*T_out + t_out)
    const uint32_t num_rows = get_arg_val<uint32_t>(2);   // valid output rows for this core

    const auto src = TensorAccessor(src_args, src_addr, C * 4);

    Noc noc;
    CircularBuffer act_cb(act_cb_id);
    CircularBuffer scalar_cb(scalar_cb_id);
    CircularBuffer scratch_cb(scratch_cb_id);

    const uint32_t num_blocks = (num_rows + BLOCK_T - 1) / BLOCK_T;
    constexpr uint32_t scalar_tile_bytes = 1024 * 4;  // fp32 32x32 tile

    // Fill the K scalar tiles once (resident, depth K); compute reads them by tap index without popping.
    fill_scalar_tiles(scalar_cb, scalar_tile_bytes);

    // Scratch is reused across blocks, so zero it once here. The channel-pad tail [C*4, C_pad*4) is
    // never written by a read, and after block 0 every slot holds defined data.
    if constexpr (coalesce) {
        zero_cb_region(noc, scratch_cb, union_sticks * padded_stick_bytes);
    }

    // The op is read-bound; the coalesced path (B==1) reads each block's union of input pages once
    // and gathers taps from L1 instead of re-reading each overlapping element per tap.
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base_row = row_start + blk * BLOCK_T;
        if constexpr (coalesce) {
            read_block_coalesced(noc, src, act_cb, scratch_cb, base_row);
        } else {
            read_block(noc, src, act_cb, blk, base_row, num_rows);
        }
    }
}
