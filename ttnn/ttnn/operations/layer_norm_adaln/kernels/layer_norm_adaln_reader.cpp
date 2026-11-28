// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_adaln — reader (NCRISC; NoC0, or NoC1 on the NOC_SWAP_EAST_COLS easternmost grid columns).
//
//   prepare_reduce_scaler_once : SUM/REDUCE_ROW scaler tile (1.0) -> cb_reduce_scaler (kernel scope)
//   load_block                 : block_rows x block_width x tiles -> cb_input_tiles, one barrier + push per tile row
//   affine (affine_in_reader: the NOC_SWAP_EAST_COLS columns, where this reader is the core's NoC1 RISC): the column's
//                                w / b row-0 fetch + multicast (layer_norm_adaln_affine.hpp); elsewhere the writer runs
//                                it
//   read_stagger (`stagger`, one-tile-row-per-core blockings whose gated phases are DRAM-bandwidth bound;
//                                see layer_norm_adaln_program_descriptor.py STAGGER_*): the row groups run in two or
//                                three phases. A gated core waits on its stagger semaphore before issuing its row; its
//                                previous-phase same-slice predecessor bumps it (raw NoC atomic) once its row has
//                                landed. The one-row blocking is a chip-wide serial chain (read wall -> stats chain ->
//                                write wall); staggering it lets earlier phases' stats chain + writes overlap later
//                                reads. No endpoint-row group (NoC1 DRAM write endpoint) is gated where an earlier
//                                phase's store burst would converge on its exchange (perf_experiments/read_stagger_v2,
//                                stagger_3phase: 256x3072 wb fp32 -7..9%; 3 phases 288..640x3072 -2.4..-6%).
//                                Deadlock-free: point-to-point, phase p waits only on phase p - 1, and a group's S
//                                members share a phase so the stats exchange never waits across phases.
//   Profiler zones hash name + file + line to 16 bits: after moving a zone, re-check collisions across the op's kernels
//   (perf_experiments/dev_tail/hashcheck.py <op dir>); the device profiler throws on a collision.
//
// Rows are assigned per group (all S slices of a group run identical block sequences).
// Each core visits its row's block_width tiles rotated by a per-core start offset (RT arg `rot`, core id % DRAM banks):
// every row starts at a page on bank 0 (tensor_w_tiles and block_width are multiples of the bank count on the common
// shapes), so unrotated, every core would hit bank 0 first and walk the banks in lockstep (perf_experiments/dram_bw).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "layer_norm_adaln_affine.hpp"

void kernel_main() {
    constexpr uint32_t block_height = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t chunk = get_compile_time_arg_val(3);  // tiles per push within a row (stats_chain streaming)
    constexpr auto input_args = TensorAccessorArgs<4>();
    // affine_fetch_fix: on the NOC_SWAP_EAST_COLS columns this reader is the core's NoC1 RISC and owns the affine
    // w / b column fetch + multicast (the writer there is on the congested NoC0). w / b accessor args follow the
    // input's (the input's stand in when unused).
    constexpr bool affine_in_reader = get_named_compile_time_arg_val("affine_in_reader") != 0;
    constexpr uint32_t affine_fetch_bytes = get_named_compile_time_arg_val("affine_fetch_bytes");
    constexpr uint32_t affine_sem_id = get_named_compile_time_arg_val("affine_sem_id");
    constexpr bool stagger = get_named_compile_time_arg_val("stagger") != 0;
    constexpr uint32_t stagger_sem_id = get_named_compile_time_arg_val("stagger_sem_id");
    [[maybe_unused]] constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr uint32_t stagger_rt = 5;  // [gated, has_successor, succ_x, succ_y] (read_stagger; zeros when off)
    constexpr uint32_t affine_rt = stagger_rt + 4;
    constexpr uint32_t num_chunks = (block_width + chunk - 1) / chunk;
    static_assert(num_chunks >= 1 && num_chunks <= 15, "one NoC read transaction id (1..15) per chunk");

    constexpr uint32_t cb_input_tiles = 0;
    constexpr uint32_t cb_reduce_scaler = 1;

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t group_rows = get_arg_val<uint32_t>(2);
    const uint32_t w_start = get_arg_val<uint32_t>(3);
    const uint32_t rot = get_arg_val<uint32_t>(4) % block_width;  // per-core start column (bank stagger)

    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    {
        MaybeDeviceZoneScope("r_scaler");
        DataflowBuffer scaler_buf(cb_reduce_scaler);
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_reduce_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>(scaler_buf);
    }

    // Injector / alone: the w / b reads are issued before the first x row (no wait) and finished (barrier, column
    // multicast, push) right after that row lands, so the fetch never delays the x loads. Receiver: waits for the flag
    // once block 0 is loaded (compute needs w / b only at block 0's combine).
    [[maybe_unused]] AffineCol aff{};
    if constexpr (affine_in_reader) {
        aff = affine_issue<block_width, affine_fetch_bytes, affine_sem_id>(weight_args, bias_args, w_start, affine_rt);
    }

    uint32_t rows_done = 0;
    for (uint32_t block_idx = 0; rows_done < group_rows; ++block_idx) {
        const uint32_t block_rows = (group_rows - rows_done) < block_height ? (group_rows - rows_done) : block_height;

        // load_block: one push per tile row of block_width tiles.
        for (uint32_t r = 0; r < block_rows; ++r) {
            MaybeDeviceZoneScope("r_load_row");
            {
                MaybeDeviceZoneScope("r_load_reserve");
                cb_reserve_back(cb_input_tiles, block_width);
            }
            const uint32_t l1 = get_write_ptr(cb_input_tiles);
            const uint32_t page0 = (row_start + rows_done + r) * tensor_w_tiles + w_start;
            if constexpr (stagger) {
                if (block_idx == 0 && r == 0 && get_arg_val<uint32_t>(stagger_rt) != 0) {
                    MaybeDeviceZoneScope("r_stagger_wait");
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(stagger_sem_id)), 1);
                }
            }
            {
                MaybeDeviceZoneScope("r_load_issue");
                uint32_t c = rot;
                for (uint32_t j = 0; j < num_chunks; ++j) {
                    if constexpr (num_chunks > 1) {
                        noc_async_read_set_trid(j + 1);
                    }
                    const uint32_t n = (block_width - j * chunk) < chunk ? (block_width - j * chunk) : chunk;
                    for (uint32_t i = 0; i < n; ++i) {
                        noc_async_read_page(page0 + c, input_accessor, l1 + c * tile_bytes);
                        c = (c + 1 == block_width) ? 0 : c + 1;
                    }
                }
            }
            {
                MaybeDeviceZoneScope("r_load_barrier");
                if constexpr (num_chunks > 1) {
                    // Push each chunk as soon as its reads (tagged trid j + 1) land: compute streams the row's stats
                    // behind the load instead of waiting for the whole row.
                    for (uint32_t j = 0; j < num_chunks; ++j) {
                        const uint32_t n = (block_width - j * chunk) < chunk ? (block_width - j * chunk) : chunk;
                        noc_async_read_barrier_with_trid(j + 1);
                        cb_push_back(cb_input_tiles, n);
                    }
                    noc_async_read_set_trid(0);
                } else {
                    noc_async_read_barrier();
                    cb_push_back(cb_input_tiles, block_width);
                }
            }
            if constexpr (stagger) {
                if (block_idx == 0 && r == 0 && get_arg_val<uint32_t>(stagger_rt + 1) != 0) {
                    // This core's row has landed: release its phase-1 successor (same slice).
                    noc_semaphore_inc(
                        get_noc_addr(
                            get_arg_val<uint32_t>(stagger_rt + 2),
                            get_arg_val<uint32_t>(stagger_rt + 3),
                            get_semaphore(stagger_sem_id)),
                        1);
                }
            }
            if constexpr (affine_in_reader) {
                if (block_idx == 0 && r == 0 && aff.role != AFFINE_RECEIVER) {
                    MaybeDeviceZoneScope("r_affine");
                    affine_finish<block_width>(aff, affine_rt);
                }
            }
        }
        rows_done += block_rows;
        // Receiver: compute consumes w / b at block 0's combine, so push them once block 0 is loaded (never blocks the
        // x loads of block 0; block 1 cannot be consumed before block 0's combine anyway).
        if constexpr (affine_in_reader) {
            if (block_idx == 0 && aff.role == AFFINE_RECEIVER) {
                MaybeDeviceZoneScope("r_affine");
                affine_finish<block_width>(aff, affine_rt);
            }
        }
    }
    if constexpr (stagger) {
        noc_async_atomic_barrier();
    }
}
