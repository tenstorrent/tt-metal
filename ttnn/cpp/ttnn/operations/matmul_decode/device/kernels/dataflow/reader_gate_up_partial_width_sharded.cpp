// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused gate+up partial-width-sharded matmul activation (in0 / A) reader.
//
// Same A-gather as reader_partial_width_sharded.cpp (the shared handshake in gather_common.hpp, run
// ONCE), but publishes BOTH resident weights -- gate_b (in1) and up_b (in1b) -- so the single
// gathered A feeds both the gate and up partial matmuls. Sharing the gather is the whole point of
// the fused op. (No gated-residual epilogue here; the gate_up output is consumed by the MLP-down.)
#include "gather_common.hpp"
using experimental::CircularBuffer;

void kernel_main() {
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(14);  // gate_b
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t in1b_cb_index = get_compile_time_arg_val(20);  // up_b (second resident weight)

    // Both weights are already resident in L1; just publish them to compute.
    CircularBuffer in1_cb(in1_cb_index);
    CircularBuffer in1b_cb(in1b_cb_index);
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);
    in1b_cb.reserve_back(in1_num_tiles);
    in1b_cb.push_back(in1_num_tiles);

    gather_full_a<21>();  // in0 accessor at slot 21 (after the extra in1b)
}
