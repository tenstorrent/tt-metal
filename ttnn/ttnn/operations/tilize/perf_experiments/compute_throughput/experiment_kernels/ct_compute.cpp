// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute — ARM SWITCH for the `compute_throughput` bake-off.
//
// Arm 0 is the op's current compute kernel, reproduced verbatim (same helper,
// same template arguments, same one-call-per-core structure).  Every other arm
// changes exactly ONE thing about how the TRISC does the work; the host,
// blocking, CB geometry, grid, dtypes and the whole ComputeConfig (fp32 dest,
// fidelity, sync mode) come from the op's own program descriptor, untouched.
//
//   0 baseline        op's current call: helper, WaitBlock
//   1 wait_upfront    helper, WaitMode::WaitUpfront (one wait for all input)
//   2 nowait          helper, WaitMode::NoWait      (handshake floor probe;
//                     only legal where the input CB is aliased on a resident
//                     shard, i.e. the data is there before the program starts)
//   3 payload_floor   raw LLK, NO CB handshake at all — pure LLK payload cost.
//                     Produces WRONG output by construction (the CB pointers
//                     never advance): a measurement, never a candidate.
//   4 wide_dest       CANDIDATE.  Fast path: byte-identical to arm 0.  Regular
//                     (non-fast) path: `tilize_block_wide<k_dest_window>`
//                     instead of `tilize_block` — a full DEST section per
//                     acquire instead of one tile.
//   5 raw_regular_ctl CONTROL for arm 4: same open-coded loop, but plain
//                     `tilize_block`.  Isolates the DEST window from the
//                     handshake rewrite.
//   6 wide_dest_w2    arm 4 with window = 2
//   7 wide_dest_half  arm 4 with window = k_dest_window / 2
//   8 wide_dest_x2    arm 4 with 2x the window — probe: is the capacity rule
//                     conservative for TINY tiles?  Correctness-gated.
//   9 wide_dest_x4    arm 4 with 4x the window — same probe.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/ttnn/operations/tilize/perf_experiments/compute_throughput/experiment_kernels/tilize_wide.hpp"

#ifndef CT_VARIANT
#define CT_VARIANT 0
#endif

namespace {

constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_output_tiles = 16;

constexpr uint32_t wt_chunk = get_compile_time_arg_val(0);
constexpr uint32_t needs_cast = get_compile_time_arg_val(1);

using namespace compute_kernel_lib::tilize_config;

// The helper's own fast/regular decision, re-evaluated here so the arms can
// branch on it at compile time (tilize_helpers.inl:66).
constexpr bool k_use_fast = compute_kernel_lib::can_use_fast_tilize<wt_chunk, cb_input_sticks, cb_output_tiles>();

// ---------------------------------------------------------------------------
// DEST tile capacity for the regular tilize path.
//
// `compute_kernel_lib::get_dest_limit()` keys the 32-bit-DEST halving on
// DST_ACCUM_MODE alone, which is the fp32-accumulation flag.  That is not the
// whole story for a datacopy: a 32-BIT INPUT DATUM (Float32/Tf32/Int32/UInt32)
// occupies a 32-bit DEST slot whether or not fp32 accumulation is on, so the
// capacity is halved for those formats too.  Measured: uint32 with
// DST_ACCUM_MODE=0 and a window of `get_dest_limit()`==8 is NOT bit-exact; a
// window of 4 is.  This is the corrected rule.
// ---------------------------------------------------------------------------
constexpr bool k_input_is_32bit = []() {
    constexpr uint32_t f = compute_kernel_lib::dfb_l1_format<cb_input_sticks>();
    return f == static_cast<uint32_t>(DataFormat::Float32) || f == static_cast<uint32_t>(DataFormat::Tf32) ||
           f == static_cast<uint32_t>(DataFormat::Int32) || f == static_cast<uint32_t>(DataFormat::UInt32);
}();

constexpr uint32_t k_dest_window = (compute_kernel_lib::get_fp32_dest_acc_enabled() || k_input_is_32bit)
                                       ? (compute_kernel_lib::get_dst_full_sync_enabled() ? 8u : 4u)
                                       : (compute_kernel_lib::get_dst_full_sync_enabled() ? 16u : 8u);

constexpr uint32_t k_window_half = (k_dest_window / 2u) ? (k_dest_window / 2u) : 1u;
constexpr uint32_t k_window_x2 = k_dest_window * 2u;
constexpr uint32_t k_window_x4 = k_dest_window * 4u;

template <WaitMode wait_mode>
ALWI void helper_call(uint32_t num_blocks) {
    if constexpr (needs_cast) {
        compute_kernel_lib::tilize<
            wt_chunk,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            wait_mode,
            ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            Fp32Mode::Fast>(num_blocks);
    } else {
        compute_kernel_lib::tilize<
            wt_chunk,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            wait_mode,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
            Fp32Mode::Fast>(num_blocks);
    }
}

// The reconfigure + init prologue the helper performs, open-coded so the raw
// arms below start from exactly the same hardware state as arm 0.
ALWI void raw_prologue() {
    if constexpr (needs_cast) {
        reconfig_data_format_srca(cb_input_sticks);
#ifndef ARCH_BLACKHOLE
        if constexpr (k_use_fast) {
            reconfig_data_format_srcb(cb_input_sticks);
        }
#endif
        pack_reconfig_data_format(cb_output_tiles);
    }
    if constexpr (k_use_fast) {
        fast_tilize_init(cb_input_sticks, wt_chunk, cb_output_tiles);
    } else {
        tilize_init(cb_input_sticks, wt_chunk, cb_output_tiles);
    }
}

ALWI void raw_epilogue() {
    if constexpr (k_use_fast) {
        fast_tilize_uninit(cb_input_sticks, cb_output_tiles, wt_chunk);
    } else {
        tilize_uninit(cb_input_sticks, cb_output_tiles);
    }
}

// Open-coded per-block loop with a pluggable block body.  `WIDE == 0` means
// "call the stock `tilize_block`" (the control arm); otherwise the DEST window.
template <uint32_t window>
ALWI void raw_regular_loop(uint32_t num_blocks) {
    raw_prologue();
    DataflowBuffer in_dfb(cb_input_sticks);
    DataflowBuffer out_dfb(cb_output_tiles);
    for (uint32_t block = 0; block < num_blocks; ++block) {
        in_dfb.wait_front(wt_chunk);
        out_dfb.reserve_back(wt_chunk);
        if constexpr (window == 0) {
            tilize_block(cb_input_sticks, wt_chunk, cb_output_tiles);
        } else {
            tilize_ct::tilize_block_wide<window>(cb_input_sticks, wt_chunk, cb_output_tiles);
        }
        out_dfb.push_back(wt_chunk);
        in_dfb.pop_front(wt_chunk);
    }
    raw_epilogue();
}

}  // namespace

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);

    if (num_blocks == 0) {
        return;
    }

    MaybeDeviceZoneScope("compute_tilize");

#if CT_VARIANT == 0
    helper_call<WaitMode::WaitBlock>(num_blocks);

#elif CT_VARIANT == 1
    helper_call<WaitMode::WaitUpfront>(num_blocks);

#elif CT_VARIANT == 2
    helper_call<WaitMode::NoWait>(num_blocks);

#elif CT_VARIANT == 3
    // Payload floor: drop the two BLOCKING halves of the handshake (wait_front /
    // reserve_back) and keep only the non-blocking pointer advance (push_back /
    // pop_front), which the writer needs in order to drain at all.  Legal only
    // where both CBs are aliased on a resident shard and sized to hold the whole
    // shard — i.e. the same-spec sharded plan this bench measures.
    {
        raw_prologue();
        DataflowBuffer in_dfb(cb_input_sticks);
        DataflowBuffer out_dfb(cb_output_tiles);
        for (uint32_t block = 0; block < num_blocks; ++block) {
            if constexpr (k_use_fast) {
                fast_tilize_block(cb_input_sticks, wt_chunk, cb_output_tiles);
            } else {
                tilize_block(cb_input_sticks, wt_chunk, cb_output_tiles);
            }
            out_dfb.push_back(wt_chunk);
            in_dfb.pop_front(wt_chunk);
        }
        raw_epilogue();
    }

#elif CT_VARIANT == 4
    if constexpr (k_use_fast) {
        helper_call<WaitMode::WaitBlock>(num_blocks);
    } else {
        raw_regular_loop<k_dest_window>(num_blocks);
    }

#elif CT_VARIANT == 5
    if constexpr (k_use_fast) {
        helper_call<WaitMode::WaitBlock>(num_blocks);
    } else {
        raw_regular_loop<0>(num_blocks);
    }

#elif CT_VARIANT == 6
    if constexpr (k_use_fast) {
        helper_call<WaitMode::WaitBlock>(num_blocks);
    } else {
        raw_regular_loop<2>(num_blocks);
    }

#elif CT_VARIANT == 7
    if constexpr (k_use_fast) {
        helper_call<WaitMode::WaitBlock>(num_blocks);
    } else {
        raw_regular_loop<k_window_half>(num_blocks);
    }

#elif CT_VARIANT == 8
    // Over-wide probe: is the DEST tile capacity actually larger for TINY tiles
    // (tile_h < 32)?  Correctness-gated — if a tiny tile still occupies a full
    // 32-row DEST slot this is illegal and the bit-exact check will say so.
    if constexpr (k_use_fast) {
        helper_call<WaitMode::WaitBlock>(num_blocks);
    } else {
        raw_regular_loop<k_window_x2>(num_blocks);
    }

#elif CT_VARIANT == 9
    if constexpr (k_use_fast) {
        helper_call<WaitMode::WaitBlock>(num_blocks);
    } else {
        raw_regular_loop<k_window_x4>(num_blocks);
    }

#else
#error "unknown CT_VARIANT"
#endif
}
