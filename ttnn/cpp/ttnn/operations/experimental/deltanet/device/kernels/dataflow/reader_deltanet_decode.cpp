// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel for fused DeltaNet decode.
//
// Per-head: reads state (Dk*Dv tiles), q/k (Dk tiles each), v (Dv tiles),
// decay g (1 tile), beta (1 tile) from DRAM into circular buffers.
// Also constructs k_T (transposed k) by element-level rearrangement in L1.
// Each core handles one head; the per-head tile offsets are runtime args.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_state    = get_compile_time_arg_val(0);
    constexpr uint32_t cb_q        = get_compile_time_arg_val(1);
    constexpr uint32_t cb_k        = get_compile_time_arg_val(2);
    constexpr uint32_t cb_v        = get_compile_time_arg_val(3);
    constexpr uint32_t cb_g        = get_compile_time_arg_val(4);
    constexpr uint32_t cb_beta     = get_compile_time_arg_val(5);
    constexpr uint32_t Dk_tiles    = get_compile_time_arg_val(6);
    constexpr uint32_t Dv_tiles    = get_compile_time_arg_val(7);
    constexpr uint32_t cb_k_T      = get_compile_time_arg_val(8);
    constexpr auto accessor_args   = TensorAccessorArgs<9>();

    const uint32_t state_addr       = get_arg_val<uint32_t>(0);
    const uint32_t q_addr           = get_arg_val<uint32_t>(1);
    const uint32_t k_addr           = get_arg_val<uint32_t>(2);
    const uint32_t v_addr           = get_arg_val<uint32_t>(3);
    const uint32_t g_addr           = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr        = get_arg_val<uint32_t>(5);
    const uint32_t state_start_tile = get_arg_val<uint32_t>(6);
    const uint32_t q_start_tile     = get_arg_val<uint32_t>(7);
    const uint32_t k_start_tile     = get_arg_val<uint32_t>(8);
    const uint32_t v_start_tile     = get_arg_val<uint32_t>(9);
    const uint32_t g_start_tile     = get_arg_val<uint32_t>(10);
    const uint32_t beta_start_tile  = get_arg_val<uint32_t>(11);

    constexpr uint32_t state_tiles = Dk_tiles * Dv_tiles;
    const uint32_t tile_bytes = get_tile_size(cb_state);

    const auto state_acc = TensorAccessor(accessor_args, state_addr, tile_bytes);
    const auto q_acc     = TensorAccessor(accessor_args, q_addr, tile_bytes);
    const auto k_acc     = TensorAccessor(accessor_args, k_addr, tile_bytes);
    const auto v_acc     = TensorAccessor(accessor_args, v_addr, tile_bytes);
    const auto g_acc     = TensorAccessor(accessor_args, g_addr, tile_bytes);
    const auto beta_acc  = TensorAccessor(accessor_args, beta_addr, tile_bytes);

    // Read state: Dk_tiles * Dv_tiles tiles for this head
    {
        cb_reserve_back(cb_state, state_tiles);
        uint32_t l1_addr = get_write_ptr(cb_state);
        for (uint32_t t = 0; t < state_tiles; t++) {
            noc_async_read_tile(state_start_tile + t, state_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_state, state_tiles);
    }

    // Read q: Dk_tiles tiles
    {
        cb_reserve_back(cb_q, Dk_tiles);
        uint32_t l1_addr = get_write_ptr(cb_q);
        for (uint32_t t = 0; t < Dk_tiles; t++) {
            noc_async_read_tile(q_start_tile + t, q_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, Dk_tiles);
    }

    // Read k: Dk_tiles tiles (save base address for transpose)
    uint32_t k_base_l1;
    {
        cb_reserve_back(cb_k, Dk_tiles);
        k_base_l1 = get_write_ptr(cb_k);
        uint32_t l1_addr = k_base_l1;
        for (uint32_t t = 0; t < Dk_tiles; t++) {
            noc_async_read_tile(k_start_tile + t, k_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_k, Dk_tiles);
    }

    // Construct k_T: transpose each k row-vector tile into a column-vector tile.
    // BF16 tile layout (2048 bytes, no header):
    //   Face 0: rows 0-15,  cols 0-15  (256 uint16 elements, offset 0)
    //   Face 1: rows 0-15,  cols 16-31 (offset 256)
    //   Face 2: rows 16-31, cols 0-15  (offset 512)
    //   Face 3: rows 16-31, cols 16-31 (offset 768)
    // Element (r,c): face = (r/16)*2 + (c/16), idx = face*256 + (r%16)*16 + (c%16)
    //
    // k row vector:  k[0,j] at face (0, j/16), idx = (j/16)*256 + j%16
    // k_T col vector: k_T[j,0] at face (j/16, 0), idx = (j/16)*2*256 + (j%16)*16
    {
        cb_reserve_back(cb_k_T, Dk_tiles);
        uint32_t k_src = k_base_l1;
        uint32_t k_T_dst = get_write_ptr(cb_k_T);

        for (uint32_t t = 0; t < Dk_tiles; t++) {
            volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_src);
            volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_T_dst);

            // Zero the destination tile (1024 uint16 elements)
            for (uint32_t i = 0; i < 1024; i++) {
                dst[i] = 0;
            }

            // k[0, 0..15] → k_T[0..15, 0]  (Face 0 row 0 → Face 0 col 0)
            for (uint32_t j = 0; j < 16; j++) {
                dst[j * 16] = src[j];
            }
            // k[0, 16..31] → k_T[16..31, 0]  (Face 1 row 0 → Face 2 col 0)
            for (uint32_t j = 0; j < 16; j++) {
                dst[512 + j * 16] = src[256 + j];
            }

            k_src += tile_bytes;
            k_T_dst += tile_bytes;
        }

        cb_push_back(cb_k_T, Dk_tiles);
    }

    // Read v: Dv_tiles tiles
    {
        cb_reserve_back(cb_v, Dv_tiles);
        uint32_t l1_addr = get_write_ptr(cb_v);
        for (uint32_t t = 0; t < Dv_tiles; t++) {
            noc_async_read_tile(v_start_tile + t, v_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, Dv_tiles);
    }

    // Read g (decay scalar): 1 tile
    {
        cb_reserve_back(cb_g, 1);
        noc_async_read_tile(g_start_tile, g_acc, get_write_ptr(cb_g));
        noc_async_read_barrier();
        cb_push_back(cb_g, 1);
    }

    // Read beta scalar: 1 tile
    {
        cb_reserve_back(cb_beta, 1);
        noc_async_read_tile(beta_start_tile, beta_acc, get_write_ptr(cb_beta));
        noc_async_read_barrier();
        cb_push_back(cb_beta, 1);
    }
}
