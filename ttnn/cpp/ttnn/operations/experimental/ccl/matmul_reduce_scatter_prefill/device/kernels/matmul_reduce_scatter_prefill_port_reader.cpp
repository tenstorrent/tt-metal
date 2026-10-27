// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter_prefill — port reader (NCRISC, NoC0): gather_unit.
//
// Port (direction d, link l) forwards, in entry order, the units (entry, core halves) whose role on this chip is
// start(d) or relay(d), restricted to its cores (link(k) == l). Per core of a unit: wait the core's mailbox slot (one
// increment per (core, unit)), NoC-read its finished sub-block (the wave tiles of the entry, dense at the front of slot
// entry mod depth_handoff of the core's cb_finished) in chunks of chunk_tiles into cb_port_chunks, then ack the core
// (consumed_d) once the read barrier covering its last chunk returned. A batch ends at a multiple of `group` chunks in
// the CB (2 group chunks, so a reservation never wraps), at the end of a unit, and before any blocking mailbox wait (no
// ack is held while waiting). At the end every mailbox slot is restored to 0 (all increments of this call have landed).
//
// Raw-API note (design "Helpers considered and rejected"): a pull with per-producer acks; no helper expresses it.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"  // permanent stage zones (free when the profiler is off)
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_args.hpp"

namespace mrs = matmul_reduce_scatter_prefill;

void kernel_main() {
    namespace ct = mrs::port_ct;
    constexpr uint32_t cb_port_chunks = get_compile_time_arg_val(ct::CB_PORT_CHUNKS);
    constexpr uint32_t P = get_compile_time_arg_val(ct::P);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(ct::CHUNK_TILES);
    constexpr uint32_t group = get_compile_time_arg_val(ct::GROUP);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(ct::TILE_BYTES);
    constexpr uint32_t depth_handoff = get_compile_time_arg_val(ct::DEPTH_HANDOFF);
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    constexpr uint32_t mailbox_stride = get_compile_time_arg_val(ct::MAILBOX_STRIDE);  // bytes per matmul core
    constexpr uint32_t max_cores = get_compile_time_arg_val(ct::MAX_CORES);    // matmul cores this port serves (bound)
    constexpr uint32_t num_waves = get_compile_time_arg_val(ct::NUM_WAVES);    // entry e is wave e mod num_waves
    constexpr uint32_t num_chunks = get_compile_time_arg_val(ct::NUM_CHUNKS);  // the unit walk repeats per chunk
    constexpr uint32_t entries_per_chunk = get_compile_time_arg_val(ct::ENTRIES_PER_CHUNK);

    // staging address (cb_finished slot 0) on every core
    const uint32_t finished_base = get_common_arg_val<uint32_t>(mrs::port_reader_common::FINISHED_BASE);
    const uint32_t mailbox_base = get_common_arg_val<uint32_t>(mrs::port_reader_common::MAILBOX_BASE);
    const uint32_t consumed_id = get_arg_val<uint32_t>(mrs::port_reader_rt::CONSUMED_ID);
    const uint32_t num_cores = get_arg_val<uint32_t>(mrs::port_reader_rt::NUM_CORES);
    const uint32_t num_units = get_arg_val<uint32_t>(mrs::port_reader_rt::NUM_UNITS);
    const uint32_t wave_tiles_idx = mrs::port_reader_rt::COUNT;  // per wave: tiles one core sends (<= P)
    const uint32_t cores_idx = wave_tiles_idx + num_waves;       // per core: x | y << 8 | half << 16 | k << 20
    const uint32_t units_idx = cores_idx + num_cores;  // per unit: entry | half_mask << 8 | slots (sender side)
    const uint32_t consumed_addr = get_semaphore(consumed_id);
    ASSERT(num_cores <= max_cores);

    uint32_t handoffs[max_cores];
    for (uint32_t i = 0; i < num_cores; ++i) {
        handoffs[i] = 0;
    }

    uint32_t pos = 0;   // chunks pushed so far (batch alignment)
    uint32_t c = 0;     // chunks in the current batch
    uint32_t need = 0;  // chunks the current reservation covers
    uint32_t wptr = 0;
    uint32_t acks[group];
    uint32_t num_acks = 0;
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(cb_port_chunks, c * chunk_tiles);
        pos += c;
        c = 0;
        for (uint32_t i = 0; i < num_acks; ++i) {
            const uint32_t cw = acks[i];
            noc_semaphore_inc(get_noc_addr(cw & 0xFF, (cw >> 8) & 0xFF, consumed_addr), 1);
        }
        num_acks = 0;
    };

    for (uint32_t ug = 0; ug < num_chunks * num_units; ++ug) {
        DeviceZoneScopedN("PORT_GATHER_UNIT");  // gather of one unit: mailbox waits + chunk reads (occupancy)
        const uint32_t u = ug % num_units;
        const uint32_t unit = get_arg_val<uint32_t>(units_idx + u);
        const uint32_t entry = unit & 0xFF;
        const uint32_t half_mask = (unit >> 8) & 0x3;
        const uint32_t first_half = (unit >> 10) & 0x1;
        // the producing core's cb_finished slot: its global entry index (chunk-major) mod the handoff depth
        const uint32_t slot_off = (((ug / num_units) * entries_per_chunk + entry) % depth_handoff) * P * tile_bytes;
        const uint32_t tiles = get_arg_val<uint32_t>(wave_tiles_idx + entry % num_waves);
        const uint32_t chunks_per_core = (tiles + chunk_tiles - 1) / chunk_tiles;
        for (uint32_t walk = 0; walk < 2 * num_cores; ++walk) {  // halves in unit order (first_half first)
            const uint32_t hh = first_half ^ (walk >= num_cores ? 1u : 0u);
            const uint32_t i = walk >= num_cores ? walk - num_cores : walk;
            const uint32_t cw = get_arg_val<uint32_t>(cores_idx + i);
            if (!((half_mask >> hh) & 1) || ((cw >> 16) & 0xF) != hh) {
                continue;
            }
            const uint32_t k = cw >> 20;
            volatile tt_l1_ptr uint32_t* mbox =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_base + k * mailbox_stride);
            const uint32_t target = ++handoffs[i];
            invalidate_l1_cache();
            if (*mbox < target && c > 0) {
                flush();  // never hold reads / acks while blocking on a producer
            }
            noc_semaphore_wait_min(mbox, target);  // per-core wait: no zone here (marker budget, see header)
            const uint64_t src = get_noc_addr(cw & 0xFF, (cw >> 8) & 0xFF, finished_base + slot_off);
            for (uint32_t ci = 0; ci < chunks_per_core; ++ci) {
                if (c == 0) {
                    need = group - pos % group;
                    cb_reserve_back(cb_port_chunks, need * chunk_tiles);
                    wptr = get_write_ptr(cb_port_chunks);
                }
                const uint32_t n = (tiles - ci * chunk_tiles) < chunk_tiles ? (tiles - ci * chunk_tiles) : chunk_tiles;
                noc_async_read(src + ci * chunk_bytes, wptr + c * chunk_bytes, n * tile_bytes);
                if (ci + 1 == chunks_per_core) {
                    acks[num_acks++] = cw;
                }
                if (++c == need) {
                    flush();
                }
            }
        }
        if (c > 0) {
            flush();  // end of unit
        }
    }
    noc_async_atomic_barrier();
    // Re-arm: every increment of this call has landed (each slot was waited to its final count).
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t k = get_arg_val<uint32_t>(cores_idx + i) >> 20;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_base + k * mailbox_stride) = 0;
    }
}
