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
// FP32MODE: Lossless is mandatory when the OUTPUT is fp32 too — fast tilize
// truncates fp32 -> tf32, which fails the bit-identity oracle. It carries two
// static_asserts the host satisfies via fp32_dest_acc_en=True and
// unpack_to_dest_mode[cb_rm_input] = UnpackToDestFp32.
//
// Refinement 4 lever 3: for a NARROWING fp32 cast (fp32 -> bf16 / bf8b) the extra
// precision cannot survive the pack anyway, so the host may select the fast path
// (`fp32_lossless == 0`). That is a host+kernel pair, not a kernel-only flag: fast
// tilize on an fp32 input requires unpack_to_dest_mode[cb_rm_input] = Default (the
// helper static_asserts both directions), so `_compute_config` flips with this CT
// arg. The kernel's own `is_fp32_input_format` guard is kept as an AND so a
// non-fp32 input can never end up asking for Lossless.
//
// COMPUTE-ONLY PROGRAM (Refinement 4 lever 1, Path B): when both CBs are aliased
// onto the resident shards the reader and writer are pure CB bookkeeping, so the
// host drops them and this is the only kernel in the program.
//   * `no_wait` — nobody publishes cb_rm_input (the bytes are already at its
//     address), so WaitMode::NoWait drops the per-block cb_wait_front. The
//     helper's cb_pop_front still runs and still walks fifo_rd_ptr across the
//     shard, which is what selects block k's rows; total pops == shard_tiles ==
//     the CB size, so llk_pop_tiles' fifo-limit LLK_ASSERTs hold with equality.
//     Nothing pops cb_tiled_output either, and it has exactly shard_tiles pages
//     against exactly shard_tiles pushes, so cb_reserve_back never blocks.
//   * `self_arm` — the counterfactual from `examples/zero_copy_fold`: one kernel
//     but with the arm/drain FOLDED onto TRISC rather than deleted. That example
//     measures the fold at 0.74x-0.95x (slower), which is exactly why the shipped
//     form deletes the handshake instead of moving it. Kept as a bench arm so the
//     distinction is measured, not argued.
//
// RESIDENT CBs (Refinement 4b): `no_wait` above removed the per-block `cb_wait_front`
// but left the other three CB calls — `cb_reserve_back` polling a credit no consumer
// decrements, `cb_push_back` publishing a credit nobody reads, `cb_pop_front` returning
// a credit nobody reads — each of the latter two paying a `TTI_STALLWAIT(STALL_THCON, .)`.
// Refinement 4 priced that at 40.5 ns/block. `resident_cb` selects the helper's
// InputBufferMode / OutputBufferMode = Resident, which drops all of it and addresses
// block k at tile index k*chunk_wt off the CB base instead (the tilize LLK's own
// `input_tile_index` / `output_tile_index` parameters). The precondition the helper
// cannot see — "nothing is on the other end of this CB" — is the host's
// `drop_reader` / `drop_writer`, the same derivation `no_wait` already rests on.
//   bit 0 = the INPUT CB is resident (no reader kernel)
//   bit 1 = the OUTPUT CB is resident (no writer kernel)
// so 3 == Path B (both sides aliased, compute-only program) and 2 == the `alias_out`
// crossover, where a reader still fills the input CB but no writer drains the output.

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
    // --- Refinement 4 -----------------------------------------------------------
    // lever 1: the compute-only Path-B program (see the header).
    constexpr uint32_t no_wait = get_compile_time_arg_val(4);
    constexpr uint32_t self_arm = get_compile_time_arg_val(5);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(6);
    // lever 3: 1 == Fp32Mode::Lossless (the fp32 -> fp32 bit-exactness contract),
    // 0 == let an fp32 input take the fast path because the output is narrower.
    constexpr uint32_t fp32_lossless = get_compile_time_arg_val(7);
    // lever 2 (MEASUREMENT ONLY, TILIZE_LEVER_IU=1): issue one tilize call per
    // chunk-block, each with its own InitAndUninit pair, instead of one call around
    // the whole loop. The output stays bit-exact (every call is fully inited), so
    // the delta against the shipped form prices `num_blocks - 1` extra config-burst
    // pairs -- i.e. the ceiling of ANY init/uninit amortisation scheme. Never on in
    // a shipped plan.
    constexpr uint32_t per_block_init = get_compile_time_arg_val(8);
    // Refinement 4b: resident-CB bitmask (bit0 = input, bit1 = output). See the header.
    constexpr uint32_t resident_cb = get_compile_time_arg_val(9);

    using namespace compute_kernel_lib::tilize_config;

    constexpr InputBufferMode input_mode = (resident_cb & 1u) ? InputBufferMode::Resident : InputBufferMode::Circular;
    constexpr OutputBufferMode output_mode =
        (resident_cb & 2u) ? OutputBufferMode::Resident : OutputBufferMode::Circular;

    constexpr ReconfigureRegisterDatatypeMode reconfig_mode =
        needs_cast ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                   : ReconfigureRegisterDatatypeMode::NoReconfigure;

    constexpr bool fp32_in = compute_kernel_lib::is_fp32_input_format<cb_rm_input>();
    constexpr Fp32Mode fp32_mode = (fp32_in && fp32_lossless) ? Fp32Mode::Lossless : Fp32Mode::Fast;

    // The three Refinement-4 branches are mutually exclusive by construction.
    // NB these assert what THIS kernel can see; it has no CT arg for the reader's
    // existence, so "no_wait implies the reader is gone" and "self_arm implies it is
    // gone" rest entirely on the host derivation in `_plan_alias` /
    // `create_program_descriptor` (`no_wait = drop_reader and not self_arm`). Getting
    // that wrong is a hang (nothing publishes the input CB) or a single-producer
    // violation (two pushers), neither of which is catchable here — flagged by
    // `ttnn-static-analyzer` and guarded on the host side by
    // `test_tilize_refinement4.py::test_no_wait_is_set_exactly_when_the_reader_is_gone`.
    static_assert(!(no_wait && self_arm), "self_arm publishes the input CB itself, so it must WAIT for it");
    static_assert(!zones || !(no_wait || self_arm), "the zone variant instruments the three-kernel program");
    // A resident INPUT CB is exactly the `no_wait` precondition ("nothing publishes it"),
    // so the two must agree — a resident input with a wait would hang on a credit nobody
    // will post, and a waiting input that is also resident would never free its pages.
    // `self_arm` drives BOTH CBs through the full protocol from this kernel, and the zone
    // variant instruments the three-kernel program, so neither may declare residency.
    static_assert(!(resident_cb & 1u) || no_wait, "a resident input CB requires no_wait");
    static_assert(!(resident_cb & 2u) || !self_arm, "self_arm drains the output CB itself, so it is not resident");
    static_assert(!resident_cb || !zones, "the zone variant instruments the three-kernel program");
    // Lever 2's measurement arm issues one 1-block call per block, so a resident tile
    // index would restart at 0 every call and rewrite block 0 num_blocks times.
    static_assert(!resident_cb || !per_block_init, "the per-block-init arm needs the circular CB modes");

    compute_kernel_hw_startup(cb_rm_input, cb_tiled_output);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    if constexpr (self_arm) {
        // `examples/zero_copy_fold`'s fold: publish the whole resident input shard
        // from this kernel (PACK thread) so the per-block wait below is satisfied.
        cb_reserve_back(cb_rm_input, shard_tiles);
        cb_push_back(cb_rm_input, shard_tiles);
    }

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
        // The ablation must reproduce the SHIPPED program's CB traffic, so it mirrors the
        // resident flags: a resident CB has no reserve/push/pop to reproduce.
        for (uint32_t block = 0; block < num_blocks; ++block) {
            if constexpr (!no_wait) {
                cb_wait_front(cb_rm_input, chunk_wt);
            }
            if constexpr (!(resident_cb & 2u)) {
                cb_reserve_back(cb_tiled_output, chunk_wt);
                cb_push_back(cb_tiled_output, chunk_wt);
            }
            if constexpr (!(resident_cb & 1u)) {
                cb_pop_front(cb_rm_input, chunk_wt);
            }
        }
    } else {
        constexpr WaitMode wait_mode = no_wait ? WaitMode::NoWait : WaitMode::WaitBlock;
        if constexpr (per_block_init) {
            // Refinement 4 lever 2's measurement arm: num_blocks fully-inited calls
            // instead of one. Bit-exact, so the delta is the config-burst price.
            // NB this arm keeps the CIRCULAR modes: each call would restart the resident
            // tile index at 0 and rewrite block 0 num_blocks times.
            for (uint32_t block = 0; block < num_blocks; ++block) {
                compute_kernel_lib::tilize<
                    chunk_wt,
                    cb_rm_input,
                    cb_tiled_output,
                    InitUninitMode::InitAndUninit,
                    wait_mode,
                    reconfig_mode,
                    fp32_mode>(1);
            }
        } else {
            compute_kernel_lib::tilize<
                chunk_wt,
                cb_rm_input,
                cb_tiled_output,
                InitUninitMode::InitAndUninit,
                wait_mode,
                reconfig_mode,
                fp32_mode,
                RemapMode::Configure,
                input_mode,
                output_mode>(num_blocks);
        }
    }

    if constexpr (self_arm) {
        // `examples/zero_copy_fold`'s fold: retire the whole resident output shard
        // from this kernel instead of from a writer.
        cb_wait_front(cb_tiled_output, shard_tiles);
        cb_pop_front(cb_tiled_output, shard_tiles);
    }
}
