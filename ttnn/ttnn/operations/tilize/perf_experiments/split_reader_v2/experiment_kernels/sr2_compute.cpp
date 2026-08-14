// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF compute kernel (`split_reader_v2`). REAL compute — the
// library tilize helper, at the op's own precision contract — is in the loop on
// every arm, which is the whole question Perf-2 asked of this idea.
//
// A split reader needs TWO input CBs (a CB must keep exactly one producer), so
// compute has to alternate between them. That alternation is the cost this
// kernel exists to price, and `force_bb` is the control that isolates it:
//
//   two_cbs 0 — one input CB
//           1 — alternate cb_in0 / cb_in1 per block
//   split   1 — alternation is by PARITY of the block index (stride-2 reader
//               split); 2 — the first n0 blocks come from cb_in0 and the rest
//               from cb_in1 (contiguous-half reader split); 3 — periodic
//               weighted interleave: (i % period) < share comes from cb_in0.
//   force_bb 1 — take the per-block back-to-back helper form EVEN with one input
//               CB. Paired with an unchanged single reader this is the ALTERNATION
//               TAX control: it differs from the op's one-call form ONLY in
//               paying the helper's per-call prologue once per block, so
//               (force_bb arm - one-call arm) is what the split's compute side
//               costs before any read is moved.
//   drain   0 — the writer RISC drains the output CB (the op's scheme)
//           1 — COMPUTE drains its own output CB (frees the writer RISC to read)
//           2 — NOBODY drains. Legal exactly when the output CB is aliased on
//               this core's resident shard: the ring then holds the core's whole
//               shard, so every cb_reserve_back is satisfied without a pop.
//
// HELPER USAGE: every arm goes through compute_kernel_lib::tilize. The
// single-CB / one-call arm is the op's exact form. The alternating arm uses the
// helper's documented back-to-back lifecycle (InitOnly / Neither / UninitOnly)
// with one call per block, which is the ONLY way to change the input DFB between
// blocks — the helper already supports it, so NO raw LLK is needed on the
// compute side of this idea.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace {

constexpr uint32_t CB_IN0 = 0;
constexpr uint32_t CB_IN1 = 1;
constexpr uint32_t CB_OUT = 16;

}  // namespace

void kernel_main() {
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(0);
    constexpr uint32_t two_cbs = get_compile_time_arg_val(1);
    constexpr uint32_t split = get_compile_time_arg_val(2);
    constexpr uint32_t drain = get_compile_time_arg_val(3);
    constexpr uint32_t force_bb = get_compile_time_arg_val(4);
    constexpr uint32_t needs_cast = get_compile_time_arg_val(5);
    constexpr uint32_t period = get_compile_time_arg_val(6);
    constexpr uint32_t share = get_compile_time_arg_val(7);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(CB_IN0, CB_OUT);

    if (num_blocks == 0) {
        return;
    }

    using namespace compute_kernel_lib::tilize_config;
    // The op's own rule: reconfigure the unpack/pack registers ONLY on a real
    // cast. Identical on every arm — the precision contract is a fixed input.
    constexpr auto RECONFIG = needs_cast ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                                         : ReconfigureRegisterDatatypeMode::NoReconfigure;

    if constexpr (!two_cbs && !force_bb && drain != 1) {
        // The op's exact form: ONE helper call for the whole core.
        compute_kernel_lib::tilize<
            wt_chunk,
            CB_IN0,
            CB_OUT,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            RECONFIG,
            Fp32Mode::Fast>(num_blocks);
        return;
    }

    // Back-to-back form: init once, one block per call, uninit once.
    compute_kernel_lib::
        tilize<wt_chunk, CB_IN0, CB_OUT, InitUninitMode::InitOnly, WaitMode::WaitBlock, RECONFIG, Fp32Mode::Fast>(0);

    for (uint32_t i = 0; i < num_blocks; ++i) {
        bool second = false;
        if constexpr (two_cbs) {
            if constexpr (split == 1) {
                second = (i & 1u) != 0u;
            } else if constexpr (split == 2) {
                second = i >= n0;
            } else {
                second = (i % period) >= share;
            }
        }
        if (second) {
            compute_kernel_lib::tilize<
                wt_chunk,
                CB_IN1,
                CB_OUT,
                InitUninitMode::Neither,
                WaitMode::WaitBlock,
                RECONFIG,
                Fp32Mode::Fast>(1);
        } else {
            compute_kernel_lib::tilize<
                wt_chunk,
                CB_IN0,
                CB_OUT,
                InitUninitMode::Neither,
                WaitMode::WaitBlock,
                RECONFIG,
                Fp32Mode::Fast>(1);
        }
        if constexpr (drain == 1) {
            cb_wait_front(CB_OUT, wt_chunk);
            cb_pop_front(CB_OUT, wt_chunk);
        }
    }

    compute_kernel_lib::
        tilize<wt_chunk, CB_IN0, CB_OUT, InitUninitMode::UninitOnly, WaitMode::WaitBlock, RECONFIG, Fp32Mode::Fast>(0);
}
