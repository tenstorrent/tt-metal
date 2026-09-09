// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute — `tilize_block`.
//
// One `compute_kernel_lib::tilize` call per BLOCK: the block's column extent is
// the FIRST template parameter (it is what `tilize_init` programs), and the
// block's row extent is the runtime `num_blocks` argument. The helper's own
// per-tile-row traversal lives inside it, under a single init/uninit pair and a
// single unpack+pack data-format reconfig per block.
//
// No raw LLK: `compute_kernel_lib::tilize` covers this phase completely,
// including fast/regular path selection, the dtype reconfig (which is where the
// value-preserving `dtype=` cast happens, at pack time) and the CB handshake.
// `total_input_pages` is deliberately omitted — both CBs carry tile-sized
// pages, i.e. the helper's symmetric mode, which is what
// `TilizeGranularity::TILE` on the reader side produces.
//
// TWO PER-CALL OVERHEADS ARE AMORTIZED ACROSS THE CORE'S BLOCK LOOP (both are
// the helper's OWN documented parameters — neither replaces it with raw LLK):
//
//  1. `InitUninitMode`. The helper's default `InitAndUninit` repays the tilize
//     LLK init + uninit on every block. A core that owns several blocks needs
//     them once: `InitOnly` on the first, `Neither` in the middle,
//     `UninitOnly` on the last — exactly the back-to-back pattern
//     tilize_helpers.hpp example 6 documents. `block_width_tiles` (the only
//     thing `tilize_init` programs) is compile-time constant across the loop,
//     and `block_row_extent` is the helper's RUNTIME argument, so a block whose
//     row extent differs from its predecessor's still reuses the same init.
//     Degenerates to `InitAndUninit` when the core owns one block.
//
//  2. `ReconfigureRegisterDatatypeMode`. The helper reconfigs unpack srcA/srcB
//     and the pack format at every call by default. In THIS kernel those
//     formats are set once, correctly, by `compute_kernel_hw_startup(cb_in,
//     cb_out)` — srcA = srcB = cb_input_rows, pack = cb_output_tiles — and
//     nothing else ever runs on the TRISCs, so every subsequent reconfig writes
//     back the value already there. That includes the casting diagonal
//     (`dtype != output_dtype`): the cast is expressed by the two CBs carrying
//     DIFFERENT formats, which the startup already programmed; it is not
//     something the per-call reconfig performs. `skip_format_reconfig` is the
//     host knob that turns those writes off; it is a knob and not a hardcoded
//     mode because a future refinement that adds a SECOND compute phase to this
//     kernel (one that reprograms srcA/pack) would have to turn it back on.

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// `tilize_block`, with the two amortization knobs resolved at compile time.
// `mode` varies per loop position, so the call site picks the instantiation.
template <
    uint32_t block_width_tiles,
    uint32_t cb_input_rows,
    uint32_t cb_output_tiles,
    bool skip_format_reconfig,
    compute_kernel_lib::tilize_config::InitUninitMode mode>
ALWI void tilize_block_op(uint32_t block_row_extent) {
    using namespace compute_kernel_lib::tilize_config;
    if constexpr (skip_format_reconfig) {
        compute_kernel_lib::tilize<
            block_width_tiles,
            cb_input_rows,
            cb_output_tiles,
            mode,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure>(block_row_extent);
    } else {
        compute_kernel_lib::tilize<block_width_tiles, cb_input_rows, cb_output_tiles, mode>(block_row_extent);
    }
}

void kernel_main() {
    using compute_kernel_lib::tilize_config::InitUninitMode;

    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(3);  // R
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr bool skip_format_reconfig = get_compile_time_arg_val(6) != 0;
    // 0 whenever no core owns more than one block — then the three extra
    // InitOnly/Neither/UninitOnly instantiations can never run, and emitting
    // them only grows the TRISC binary the dispatcher has to ship. Measured:
    // that growth is worth ~4% of the wall on a ~4 us kernel, which is more
    // than the amortization can ever return at one block per core.
    constexpr bool amortize_init = get_compile_time_arg_val(7) != 0;

    const uint32_t start_block_id = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(1);
    // 1 on the solved plan (contiguous block ranges); the core count on the
    // shard-driven plan, where core i owns shards {i, i+N, i+2N, ...}.
    const uint32_t block_stride = get_arg_val<uint32_t>(2);

    // PREREQUISITE of compute_kernel_lib::tilize (tilize_helpers.hpp:89-93).
    // Also what makes `skip_format_reconfig` correct: this call is what programs
    // unpack srcA/srcB from cb_input_rows and the pack format from
    // cb_output_tiles, for the whole kernel.
    compute_kernel_hw_startup(cb_input_rows, cb_output_tiles);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        // resolve_block — the identical derivation the reader and writer run.
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        // tilize_block: block_width_tiles (CT) x block_row_extent (RT) tiles.
        // The init/uninit pair is paid once for the whole loop, not once per block.
        if constexpr (!amortize_init) {
            tilize_block_op<
                block_width_tiles,
                cb_input_rows,
                cb_output_tiles,
                skip_format_reconfig,
                InitUninitMode::InitAndUninit>(block_row_extent);
            continue;
        }
        const bool first = (b == 0);
        const bool last = (b + 1 == num_blocks_this_core);
        if (first && last) {
            tilize_block_op<
                block_width_tiles,
                cb_input_rows,
                cb_output_tiles,
                skip_format_reconfig,
                InitUninitMode::InitAndUninit>(block_row_extent);
        } else if (first) {
            tilize_block_op<
                block_width_tiles,
                cb_input_rows,
                cb_output_tiles,
                skip_format_reconfig,
                InitUninitMode::InitOnly>(block_row_extent);
        } else if (last) {
            tilize_block_op<
                block_width_tiles,
                cb_input_rows,
                cb_output_tiles,
                skip_format_reconfig,
                InitUninitMode::UninitOnly>(block_row_extent);
        } else {
            tilize_block_op<
                block_width_tiles,
                cb_input_rows,
                cb_output_tiles,
                skip_format_reconfig,
                InitUninitMode::Neither>(block_row_extent);
        }
    }
}
