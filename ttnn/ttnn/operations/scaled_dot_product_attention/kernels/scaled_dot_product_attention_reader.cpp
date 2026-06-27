// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for scaled_dot_product_attention (Flash Attention).
//
// Stage 0 (init): Initializes the running-state CBs with constant fills:
//   cb_max_old (27) ← -inf tiles (B_q_t tiles, running max m_i)
//   cb_sum_old (30) ← 0.0   tiles (B_q_t tiles, running sum l_i)
//   cb_o       (16) ← 0.0   tiles (B_q_t * D_t tiles, running output O_i)
//
// The full reader (streaming Q/K/V/mask blocks, scaler + scale-factor tiles)
// is added incrementally by later stages. Stage 0 only needs to materialize
// the initial running state so the compute kernel's recurrence has a
// correct starting point.
//
// Constant-fill pattern: cb_reserve_back → reinterpret_cast write loop →
// cb_push_back (same shape as ttnn full op's writer_full). For 0.0 the NoC
// async_write_zeros shortcut is used; for -inf the bf16 bit pattern
// (0xFF80) is written element-wise.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// CB indices (match op_design.md CB layout).
constexpr uint32_t cb_o = tt::CBIndex::c_16;
constexpr uint32_t cb_max_old = 27;
constexpr uint32_t cb_sum_old = 30;

// bf16 representation of -inf (0xFF80). See bfloat16.h: NEG_INF_BFLOAT16.
constexpr uint16_t NEG_INF_BFLOAT16 = 0xFF80;

// Fill an entire CB tile (bf16, 32x32 = 1024 elements) with a constant
// uint16 bit pattern. Used for -inf. Reserve/write/push are the caller's
// responsibility for the back-of-CB DPRINT window.
inline void fill_bf16_tile_with_const(uint32_t cb_id, uint16_t val_bits) {
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {  // 32 * 32 bf16 elements per tile
        ptr[i] = val_bits;
    }
}

// Zero-fill a CB tile (bf16). 0.0 is a special case: the NoC engine can
// zero a whole tile faster than a CPU loop.
inline void fill_bf16_tile_zero(uint32_t cb_id, uint32_t tile_bytes) {
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    // 0.0 bf16 == 0x0000; a plain zero-fill loop is simplest and matches
    // the const-fill path. (NoC async_write_zeros would also work but
    // requires a CircularBuffer wrapper.)
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
}

void kernel_main() {
    // Compile-time args (from program descriptor).
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0);       // Q-block tile rows (4)
    constexpr uint32_t D_t = get_compile_time_arg_val(1);         // head-dim tiles (D/32)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);  // bf16 tile = 2048

    // --- Stage 0: initialize running state ---

    // cb_max_old: B_q_t tiles of -inf (running max m_i = -inf).
    for (uint32_t t = 0; t < B_q_t; ++t) {
        cb_reserve_back(cb_max_old, 1);
        fill_bf16_tile_with_const(cb_max_old, NEG_INF_BFLOAT16);

        // Stage 0 checkpoint: DPRINT cb_max_old tile 0, first 4x4 — expect -inf.
        // Back of CB: between cb_reserve_back and cb_push_back.
        if (t == 0) {
            SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
            DPRINT("stage_0 init:\n{}\n", TileSlice(cb_max_old, 0, sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, false));
        }

        cb_push_back(cb_max_old, 1);
    }

    // cb_sum_old: B_q_t tiles of 0.0 (running sum l_i = 0).
    for (uint32_t t = 0; t < B_q_t; ++t) {
        cb_reserve_back(cb_sum_old, 1);
        fill_bf16_tile_zero(cb_sum_old, tile_bytes);
        cb_push_back(cb_sum_old, 1);
    }

    // cb_o: B_q_t * D_t tiles of 0.0 (running output O_i = 0).
    constexpr uint32_t num_o_tiles = B_q_t * D_t;
    for (uint32_t t = 0; t < num_o_tiles; ++t) {
        cb_reserve_back(cb_o, 1);
        fill_bf16_tile_zero(cb_o, tile_bytes);
        cb_push_back(cb_o, 1);
    }
}
