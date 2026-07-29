// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute (TRISC0/1/2). Single compute phase: RM -> TILE.
//
// One compute_kernel_lib::tilize call covers all num_blocks blocks of
// 1 x chunk_wt tiles. The helper owns wait_front / reserve_back / LLK /
// push_back / pop_front per block and issues tilize_init / tilize_uninit once
// around the whole loop (InitAndUninit).
//
// RECONFIG: NoReconfigure on the no-cast path. compute_kernel_hw_startup(
// cb_rm_input, cb_tiled_output) has already programmed srcA/srcB from the input
// CB and the packer from the output CB, so a reconfigure would be pure
// redundant CFG traffic (dominant on ~1 us sharded cases).
//
// FP32MODE: decided inside the kernel from the CB format, no CT arg needed.
// Lossless is mandatory for fp32 input — fast tilize truncates fp32 -> tf32,
// which fails the bit-identity oracle. Lossless carries two static_asserts the
// host satisfies via fp32_dest_acc_en=True and
// unpack_to_dest_mode[cb_rm_input] = UnpackToDestFp32.

// ZONES (Refinement 3b lever 1, bench only): an instrumented copy of the same
// loop with the per-block `cb_wait_front` hoisted OUT of the helper
// (WaitMode::NoWait) so it can carry its own Tracy zone. That is the one thing
// no ablation variant can measure — whether TRISC is blocked on the reads
// (CP-WAIT large) or the reads are blocked on TRISC (the reader's RD-RESV
// large). The init/uninit pair is chained across the per-block calls
// (InitOnly / Neither / UninitOnly), so the instrumented loop issues exactly the
// same 2 config bursts per kernel that the shipped one does.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_input = 0;
    constexpr uint32_t cb_tiled_output = 16;

    constexpr uint32_t chunk_wt = get_compile_time_arg_val(0);
    constexpr uint32_t needs_cast = get_compile_time_arg_val(1);
    // Perf-ablation only (TILIZE_SKIP_COMPUTE=1): drop the tilize LLK payload while
    // reproducing the helper's exact CB dance (wait chunk_wt / reserve chunk_wt /
    // push / pop, num_blocks times), so /perf-measure can attribute time to the
    // compute stage. Never set on a correctness run.
    constexpr uint32_t skip_compute = get_compile_time_arg_val(2);
    // Refinement 3b lever 1: per-RISC Tracy timeline (bench only, see the header).
    constexpr uint32_t zones = get_compile_time_arg_val(3);

    using namespace compute_kernel_lib::tilize_config;

    constexpr ReconfigureRegisterDatatypeMode reconfig_mode =
        needs_cast ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                   : ReconfigureRegisterDatatypeMode::NoReconfigure;

    constexpr Fp32Mode fp32_mode =
        compute_kernel_lib::is_fp32_input_format<cb_rm_input>() ? Fp32Mode::Lossless : Fp32Mode::Fast;

    compute_kernel_hw_startup(cb_rm_input, cb_tiled_output);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    if constexpr (zones) {
        using compute_kernel_lib::tilize;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            {
                DeviceZoneScopedN("CP-WAIT");
                cb_wait_front(cb_rm_input, chunk_wt);
            }
            {
                // The helper still owns reserve_back / LLK / push_back / pop_front;
                // only the wait moved out (WaitMode::NoWait suppresses exactly that
                // one call). One block per call, with the init/uninit chained.
                DeviceZoneScopedN("CP-LLK");
                constexpr auto kNoWait = WaitMode::NoWait;
                if (num_blocks == 1) {
                    tilize<
                        chunk_wt,
                        cb_rm_input,
                        cb_tiled_output,
                        InitUninitMode::InitAndUninit,
                        kNoWait,
                        reconfig_mode,
                        fp32_mode>(1);
                } else if (block == 0) {
                    tilize<
                        chunk_wt,
                        cb_rm_input,
                        cb_tiled_output,
                        InitUninitMode::InitOnly,
                        kNoWait,
                        reconfig_mode,
                        fp32_mode>(1);
                } else if (block + 1 == num_blocks) {
                    tilize<
                        chunk_wt,
                        cb_rm_input,
                        cb_tiled_output,
                        InitUninitMode::UninitOnly,
                        kNoWait,
                        reconfig_mode,
                        fp32_mode>(1);
                } else {
                    tilize<
                        chunk_wt,
                        cb_rm_input,
                        cb_tiled_output,
                        InitUninitMode::Neither,
                        kNoWait,
                        reconfig_mode,
                        fp32_mode>(1);
                }
            }
        }
    } else if constexpr (skip_compute) {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_wait_front(cb_rm_input, chunk_wt);
            cb_reserve_back(cb_tiled_output, chunk_wt);
            cb_push_back(cb_tiled_output, chunk_wt);
            cb_pop_front(cb_rm_input, chunk_wt);
        }
    } else {
        compute_kernel_lib::tilize<
            chunk_wt,
            cb_rm_input,
            cb_tiled_output,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            reconfig_mode,
            fp32_mode>(num_blocks);
    }
}
