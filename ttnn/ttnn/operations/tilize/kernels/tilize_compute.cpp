// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute — the single compute phase, entirely helper-driven.
//
// `block_width_tiles` IS the block-factor knob (tilize_helpers.hpp:188): two
// instantiations, one at WT_BLOCK for the full-width column-blocks and one at
// WT_TAIL for the tail column-block. Because a core's contiguous block range
// crosses the full/tail boundary at most once, the two runtime counts
// (n_full, n_tail) cover it with no per-core kernel variant.
//
// Knob settings, each a decision (op_design.md §7.1):
//   init_uninit_mode = InitAndUninit on BOTH calls — the two calls use different
//       block_width_tiles and tilize_init takes the width, so each needs its own
//       init.
//   wait_mode        = WaitBlock — per-block wait is what lets the reader run
//       ahead of compute; WaitUpfront would serialize the core behind the reader.
//   reconfig_mode    = NoReconfigure when there is nothing to cast (the
//       reconfigure exists only to drive a dtype= cast and is otherwise a fixed
//       ~150 ns waste), UnpackAndPackReconfigure on a real cast.
//   fp32_mode        = Fast — tilize's consumers are FPU ops, which re-truncate
//       to tf32 anyway; Lossless would only cost a slower path.

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t wt_block = get_compile_time_arg_val(2);
    constexpr uint32_t wt_tail = get_compile_time_arg_val(3);
    constexpr bool needs_cast = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t stub_compute = get_compile_time_arg_val(5);  // ablation (0 = off)
    constexpr uint32_t fold_resident = get_compile_time_arg_val(6);  // lever R6/C14-2 (1 = on)

    const uint32_t n_full = get_arg_val<uint32_t>(0);
    const uint32_t n_tail = get_arg_val<uint32_t>(1);
    const uint32_t fold_pages = get_arg_val<uint32_t>(2);  // R6: pages on each aliased CB

    using namespace compute_kernel_lib::tilize_config;
    constexpr ReconfigureRegisterDatatypeMode reconfig_mode =
        needs_cast ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                   : ReconfigureRegisterDatatypeMode::NoReconfigure;

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);

    // R6 (master.md C14, SECOND degree): the FOLD. On the same-spec resident
    // path both CBs are aliased onto this core's own L1 shards, so the reader
    // and writer move zero bytes and exist only to run the CB handshake —
    // `cb_reserve_back + cb_push_back` once on the input, `cb_wait_front +
    // cb_pop_front` once on the output. `fold_resident == 1` drops those two
    // kernels from the program entirely and takes the handshake here instead.
    //
    // The two halves land on DIFFERENT compute threads, which is what makes this
    // legal rather than a self-deadlock: `cb_reserve_back`/`cb_push_back` are
    // PACK-thread ops and `cb_wait_front`/`cb_pop_front` are UNPACK-thread ops
    // (compute_kernel_api/cb_api.h), so the arm below is issued by the same
    // thread that packs the tiles and the drain by the same thread that unpacks
    // them — exactly the producer/consumer split the two dataflow RISCs had.
    if constexpr (fold_resident == 1) {
        cb_reserve_back(cb_input_sticks, fold_pages);
        cb_push_back(cb_input_sticks, fold_pages);
    }

    // /perf-measure ablation arm: keep the CB reserve/push/wait/pop scaffolding
    // and the block trip counts, drop ONLY the tilize math, so the duration diff
    // classifies the op as DM-bound vs compute-bound. Not a production path;
    // `stub_compute == 0` emits the helper calls below and nothing else.
    if constexpr (stub_compute == 1) {
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const uint32_t blocks = (pass == 0) ? n_full : n_tail;
            const uint32_t w = (pass == 0) ? wt_block : wt_tail;
            for (uint32_t block = 0; block < blocks; ++block) {
                cb_wait_front(cb_input_sticks, w);
                cb_reserve_back(cb_output_tiles, w);
                cb_push_back(cb_output_tiles, w);
                cb_pop_front(cb_input_sticks, w);
            }
        }
        if constexpr (fold_resident == 1) {
            cb_wait_front(cb_output_tiles, fold_pages);
            cb_pop_front(cb_output_tiles, fold_pages);
        }
        return;
    }

    // `tilize` ASSERTs num_blocks > 0, so both calls are guarded.
    if (n_full > 0) {
        compute_kernel_lib::tilize<
            wt_block,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            reconfig_mode,
            Fp32Mode::Fast>(n_full);
    }
    if (n_tail > 0) {
        compute_kernel_lib::tilize<
            wt_tail,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            reconfig_mode,
            Fp32Mode::Fast>(n_tail);
    }

    if constexpr (fold_resident == 1) {
        cb_wait_front(cb_output_tiles, fold_pages);
        cb_pop_front(cb_output_tiles, fold_pages);
    }
}
