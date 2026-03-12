// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// BUG REPRODUCER: fast_tilize → pack_untilize corrupts face 2/3 for fp32+Wt>1
// ============================================================================
//
// Symptom:  Rows 16-31 (faces 2,3) contain garbage after round-trip.
//           Rows 0-15  (faces 0,1) are correct.
//
// Trigger:  ALL THREE conditions must be met:
//   1. fp32 data format
//   2. Wt >= 2 (multi-tile width, e.g. shape 1,1,32,64)
//   3. compute_kernel_hw_startup with srcB = bfloat16 CB (mixed-format init)
//
// Toggle:   Change USE_FAST_TILIZE below.
//             1 → fast_tilize  (BUG  — faces 2/3 corrupted)
//             0 → regular tilize (OK — all faces correct)
//
// Run:
//   scripts/tt-test.sh tests/ttnn/unit_tests/operations/rms_norm/test_fast_tilize_repro.py -v -s
//
// Notes:
//   - bf16 is NOT affected
//   - Wt=1 is NOT affected
//   - srcB = fp32 CB (same format as srcA) does NOT trigger the bug
//   - Regular tilize + pack_untilize works even with bfloat16 srcB
//   → Conclusion: fast_tilize_uninit does not fully restore HW state (set by
//     the bfloat16 srcB initial config) that pack_untilize needs for fp32+Wt>1.
//     Likely a register (Read_32b_data, dest offsets, or ADCXX) is left in a
//     state incompatible with the pack untilize face 2/3 addressing.
// ============================================================================

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"

#define USE_FAST_TILIZE 1  // <<< TOGGLE: 1 = fast_tilize (BUG), 0 = regular tilize (OK)

constexpr uint32_t c_in = 0;      // RM input sticks  (page_size = tile_size)
constexpr uint32_t c_til = 1;     // tilized tiles     (page_size = tile_size)
constexpr uint32_t c_scaler = 2;  // bfloat16 scaler   (reader fills, we never use it)
constexpr uint32_t c_out = 17;    // untilized output  (page_size = tile_size)

// Compile-time args (matches rms_norm reader/writer interface)
constexpr uint32_t Ht_max = get_compile_time_arg_val(0);  // unused, for interface compat
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t input_is_rm = get_compile_time_arg_val(2);  // always 1 for this repro
// arg 3 (has_gamma) unused

void kernel_main() {
    uint32_t Ht = get_arg_val<uint32_t>(0);
    if (Ht == 0) {
        return;
    }

    // HW startup: configures unpack/math/pack threads
    // NOTE: srcB = c_scaler (bfloat16) is critical to reproduce the bug.
    // Using srcB = c_til (float32) does NOT trigger the bug.
    // The bfloat16 srcB initial config interacts with fast_tilize_uninit
    // to leave state that corrupts pack_untilize for fp32.
    compute_kernel_hw_startup(c_til, c_scaler, c_out);

    for (uint32_t row = 0; row < Ht; ++row) {
        // ---- TILIZE: c_in (RM sticks) → c_til (tile format) ----
        cb_wait_front(c_in, Wt);
        cb_reserve_back(c_til, Wt);

#if USE_FAST_TILIZE
        fast_tilize_init(c_in, Wt, c_til);
        fast_tilize_block(c_in, Wt, c_til);
        fast_tilize_uninit(c_in, c_til);
#else
        tilize_init(c_in, Wt, c_til);
        tilize_block(c_in, Wt, c_til);
        tilize_uninit(c_in, c_til);
#endif

        cb_push_back(c_til, Wt);
        cb_pop_front(c_in, Wt);

        // ---- UNTILIZE: c_til (tile format) → c_out (RM sticks) ----
        pack_untilize_init<Wt>(c_til, c_out);

        cb_wait_front(c_til, Wt);
        cb_reserve_back(c_out, Wt);
        pack_untilize_block<Wt>(c_til, 1, c_out, 0);
        cb_pop_front(c_til, Wt);
        cb_push_back(c_out, Wt);

        pack_untilize_uninit(c_out);
    }
}
