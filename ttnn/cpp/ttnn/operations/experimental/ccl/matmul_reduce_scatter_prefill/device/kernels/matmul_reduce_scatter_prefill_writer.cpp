// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter_prefill — matmul-core writer (BRISC, NoC1): handoff_or_store_block per finished entry, in
// order.
//
// cb_finished is backed by the staging tensor (same L1 address on every core and chip); entry e of this core sits in
// slot e mod depth_handoff. An entry is one wave of a block: its rows x cols tiles are dense at the front of the slot.
// start / relay: one mailbox increment to the port (direction, link of this core); the slot is popped (FIFO) once that
// port acks (program-local consumed_fwd / consumed_bwd). fin: the valid tiles go to the output with one write barrier.
// Signalling never waits on consumption: a pop is forced only when every slot is outstanding.
//
// Raw-API note (design "Helpers considered and rejected"): the handoff is many-to-one with per-producer acks; no
// helper expresses it.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"  // permanent stage zones (free when the profiler is off)
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_args.hpp"

namespace mrs = matmul_reduce_scatter_prefill;

void kernel_main() {
    namespace ct = mrs::writer_ct;
    constexpr uint32_t cb_finished = get_compile_time_arg_val(ct::CB_FINISHED);
    constexpr uint32_t P = get_compile_time_arg_val(ct::P);
    constexpr uint32_t num_waves = get_compile_time_arg_val(ct::NUM_WAVES);  // entry e is wave e mod num_waves
    constexpr uint32_t out_n_tiles = get_compile_time_arg_val(ct::OUT_N_TILES);
    constexpr uint32_t depth_handoff = get_compile_time_arg_val(ct::DEPTH_HANDOFF);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(ct::TILE_BYTES);
    constexpr uint32_t num_entries = get_compile_time_arg_val(ct::NUM_ENTRIES);  // G x num_waves
    constexpr uint32_t cb_zero = get_compile_time_arg_val(ct::CB_ZERO);          // compute's fused-finish zero tile
    // chunks: the entry walk runs once per output chunk; chunk c covers the tiles at (i_m * chunk_m, i_n * chunk_n)
    constexpr uint32_t num_chunks = get_compile_time_arg_val(ct::NUM_CHUNKS);
    constexpr uint32_t n_inner = get_compile_time_arg_val(ct::N_INNER);
    constexpr bool m_outer = get_compile_time_arg_val(ct::M_OUTER) != 0;
    constexpr uint32_t chunk_m = get_compile_time_arg_val(ct::CHUNK_M);
    constexpr uint32_t chunk_n = get_compile_time_arg_val(ct::CHUNK_N);
    constexpr uint32_t block_m = get_compile_time_arg_val(ct::BLOCK_M);
    constexpr uint32_t block_n = get_compile_time_arg_val(ct::BLOCK_N);
    constexpr auto out_args = TensorAccessorArgs<ct::COUNT>();
    constexpr uint32_t KIND_FWD = mrs::kind::FWD, KIND_BWD = mrs::kind::BWD, KIND_FIN = mrs::kind::FIN;
    constexpr uint32_t consumed_fwd_id = mrs::sem::CONSUMED_FWD, consumed_bwd_id = mrs::sem::CONSUMED_BWD;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(mrs::writer_common::OUT_ADDR);
    // chunk-local origin of my per-core block
    const uint32_t row0 = get_arg_val<uint32_t>(mrs::writer_rt::ROW0);
    const uint32_t col0 = get_arg_val<uint32_t>(mrs::writer_rt::COL0);
    const uint32_t fwd_port_xy = get_arg_val<uint32_t>(mrs::writer_rt::FWD_PORT_XY);
    const uint32_t bwd_port_xy = get_arg_val<uint32_t>(mrs::writer_rt::BWD_PORT_XY);
    const uint32_t mailbox_slot_addr = get_common_arg_val<uint32_t>(mrs::writer_common::MAILBOX_ADDR) +
                                       get_arg_val<uint32_t>(mrs::writer_rt::MAILBOX_SLOT_OFFSET);
    const uint32_t waves_idx = mrs::writer_rt::COUNT;  // per wave: row_off | col_off << 8 | rows << 16 | cols << 24
    const uint32_t kinds_idx = waves_idx + num_waves;
    auto wave_valid = [](uint32_t valid, uint32_t off, uint32_t ext) {
        return valid <= off ? 0u : (valid - off < ext ? valid - off : ext);
    };
    // my valid tiles past `off` in chunk ci of an axis (wave_valid caps them at a wave's extent)
    auto valid_ext = [](uint32_t blk, uint32_t chunk, uint32_t ci, uint32_t off) {
        const uint32_t c0 = ci * chunk;
        const uint32_t ext = blk - c0 < chunk ? blk - c0 : chunk;
        return ext <= off ? 0u : ext - off;
    };

    {  // one bf16 zero tile: compute's fused finish adds the hold as DEST += hold + 0 (never popped)
        cb_reserve_back(cb_zero, 1);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_zero));
        for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
            z[i] = 0;
        }
        cb_push_back(cb_zero, 1);
    }
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    volatile tt_l1_ptr uint32_t* consumed_fwd =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(consumed_fwd_id));
    volatile tt_l1_ptr uint32_t* consumed_bwd =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(consumed_bwd_id));
    const uint64_t fwd_mailbox = get_noc_addr(fwd_port_xy & 0xFFFF, fwd_port_xy >> 16, mailbox_slot_addr);
    const uint64_t bwd_mailbox = get_noc_addr(bwd_port_xy & 0xFFFF, bwd_port_xy >> 16, mailbox_slot_addr);
    const uint32_t finished_base = get_read_ptr(cb_finished);  // slot 0 (read pointer is at the base at launch)

    uint32_t queue[depth_handoff];
    uint32_t head = 0, outstanding = 0, acked_fwd = 0, acked_bwd = 0;
    auto pop_oldest = [&]() {
        DeviceZoneScopedN("ACK_WAIT");
        const uint32_t kind = queue[head];
        head = head + 1 == depth_handoff ? 0 : head + 1;
        if (kind == KIND_FWD) {
            noc_semaphore_wait_min(consumed_fwd, ++acked_fwd);
        } else if (kind == KIND_BWD) {
            noc_semaphore_wait_min(consumed_bwd, ++acked_bwd);
        }
        cb_pop_front(cb_finished, P);
        --outstanding;
    };

    for (uint32_t eg = 0; eg < num_chunks * num_entries; ++eg) {
        const uint32_t e = eg % num_entries;
        const uint32_t kind = get_arg_val<uint32_t>(kinds_idx + e);
        const uint32_t slot = finished_base + (eg % depth_handoff) * P * tile_bytes;
        if (kind == KIND_FIN) {
            // stream the output: write each valid tile as soon as the finish has packed it, so the DRAM writes of
            // every core overlap the finish adds instead of all landing after them (the call's tail)
            const uint32_t c = eg / num_entries;
            const uint32_t im = m_outer ? c / n_inner : c % n_inner;
            const uint32_t in = m_outer ? c % n_inner : c / n_inner;
            const uint32_t wave = get_arg_val<uint32_t>(waves_idx + e % num_waves);
            const uint32_t row_off = wave & 0xFF, col_off = (wave >> 8) & 0xFF, cols = wave >> 24;
            const uint32_t m_valid_w = wave_valid(valid_ext(block_m, chunk_m, im, row0), row_off, (wave >> 16) & 0xFF);
            const uint32_t n_valid_w = wave_valid(valid_ext(block_n, chunk_n, in, col0), col_off, cols);
            for (uint32_t i = 0; i < m_valid_w; ++i) {
                const uint32_t page = (im * chunk_m + row0 + row_off + i) * out_n_tiles + in * chunk_n + col0 + col_off;
                {
                    DeviceZoneScopedN("FIN_WAIT");
                    cb_wait_front(cb_finished, outstanding * P + i * cols + n_valid_w);
                }
                for (uint32_t j = 0; j < n_valid_w; ++j) {
                    noc_async_write(slot + (i * cols + j) * tile_bytes, out.get_noc_addr(page + j), tile_bytes);
                }
            }
            cb_wait_front(cb_finished, (outstanding + 1) * P);
            DeviceZoneScopedN("WR_BARRIER");
            noc_async_write_barrier();
        } else {
            {
                DeviceZoneScopedN("FIN_WAIT");
                cb_wait_front(cb_finished, (outstanding + 1) * P);
            }
            noc_semaphore_inc(kind == KIND_FWD ? fwd_mailbox : bwd_mailbox, 1);
        }
        queue[(head + outstanding) % depth_handoff] = kind;
        ++outstanding;
        if (outstanding == depth_handoff) {
            pop_oldest();
        }
    }
    while (outstanding > 0) {
        pop_oldest();
    }
    noc_async_atomic_barrier();
}
