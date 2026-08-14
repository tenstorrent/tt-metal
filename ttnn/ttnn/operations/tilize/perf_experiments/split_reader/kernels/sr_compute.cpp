// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF compute kernel for design lamp L4 (`split_reader`).
//
// A split reader needs TWO input CBs (a CB must keep exactly one producer), so
// compute has to alternate between them. That is the whole scheme change; this
// kernel is the compute half of it.
//
//   two_cbs 0 — one input CB, the op's current scheme
//           1 — alternate cb_in0 / cb_in1 per block
//   split   1 — alternation is by PARITY of the block index (stride-2 reader
//               split); 2 — the first n0 blocks come from cb_in0 and the rest
//               from cb_in1 (contiguous-half reader split)
//   drain   0 — the writer RISC drains the output CB (the op's scheme)
//           1 — COMPUTE drains its own output CB (frees the writer RISC to read)
//           2 — NOBODY drains. Legal exactly when the output CB is aliased on
//               this core's resident shard: the ring then holds the core's whole
//               shard, so every cb_reserve_back is satisfied without a pop. This
//               is the cheapest way to free the second RISC on a
//               destination-local plan.
//
// HELPER USAGE: every arm goes through compute_kernel_lib::tilize. The
// single-CB / no-drain arm is the op's exact one-call form. The alternating arm
// uses the helper's documented back-to-back lifecycle (InitOnly / Neither /
// UninitOnly) with one call per block, which is the ONLY way to change the
// input DFB between blocks — the helper already supports it, so no raw LLK is
// needed here.

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

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(CB_IN0, CB_OUT);

    if (num_blocks == 0) {
        return;
    }

    using namespace compute_kernel_lib::tilize_config;

    if constexpr (!two_cbs && drain != 1) {
        // The op's exact form: ONE helper call for the whole core.
        compute_kernel_lib::tilize<
            wt_chunk,
            CB_IN0,
            CB_OUT,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
            Fp32Mode::Fast>(num_blocks);
        return;
    }

    // Back-to-back form: init once, one block per call, uninit once.
    compute_kernel_lib::tilize<
        wt_chunk,
        CB_IN0,
        CB_OUT,
        InitUninitMode::InitOnly,
        WaitMode::WaitBlock,
        ReconfigureRegisterDatatypeMode::NoReconfigure,
        Fp32Mode::Fast>(0);

    for (uint32_t i = 0; i < num_blocks; ++i) {
        bool second = false;
        if constexpr (two_cbs) {
            second = (split == 1) ? ((i & 1u) != 0u) : (i >= n0);
        }
        if (second) {
            compute_kernel_lib::tilize<
                wt_chunk,
                CB_IN1,
                CB_OUT,
                InitUninitMode::Neither,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::NoReconfigure,
                Fp32Mode::Fast>(1);
        } else {
            compute_kernel_lib::tilize<
                wt_chunk,
                CB_IN0,
                CB_OUT,
                InitUninitMode::Neither,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::NoReconfigure,
                Fp32Mode::Fast>(1);
        }
        if constexpr (drain == 1) {
            cb_wait_front(CB_OUT, wt_chunk);
            cb_pop_front(CB_OUT, wt_chunk);
        }
    }

    compute_kernel_lib::tilize<
        wt_chunk,
        CB_IN0,
        CB_OUT,
        InitUninitMode::UninitOnly,
        WaitMode::WaitBlock,
        ReconfigureRegisterDatatypeMode::NoReconfigure,
        Fp32Mode::Fast>(0);
}
