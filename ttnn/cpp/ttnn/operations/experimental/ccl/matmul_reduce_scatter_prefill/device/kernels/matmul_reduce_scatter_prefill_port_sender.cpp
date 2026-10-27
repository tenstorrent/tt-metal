// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter_prefill — port sender (BRISC, NoC1): fence_ready + send_unit (adapted from
// fabric_reduce_scatter's port sender).
//
// fence_ready: tell the peer's opposite port (which writes into this chip) that this chip has started the call, then,
// if this port sends anything, wait for the same word from the downstream chip and consume it.
// send_unit: every chunk of cb_port_chunks (in the reader's order) is one fabric packet into the downstream matmul
// core's zone slot (same logical core, same staging address on every chip); a core's last chunk carries the fused
// increment of that core's arrival counter (one ordered sender per counter: this port). Headers rotate over `group`
// pre-routed headers; flush once per batch, popping in the reader's batch boundaries.
// Chunks: the unit walk repeats once per output chunk and the downstream zone slots are a ring of zone_sets chunks'
// arrivals, so before the first unit of chunk c >= zone_sets this port waits for the downstream's credit "chunk
// c - zone_sets consumed". Symmetrically it forwards, upstream to the peer it sends to, one credit per chunk this
// chip's matmul cores (its link) consumed of that peer's arrivals (they report to the program semaphore `freed`). Every
// wait polls the forwarding, so two ports waiting on each other's credits always make progress.
//
// Raw-API note (design "Helpers considered and rejected"): the CCL helper library is unmeasured and routes by 1D hop
// count with a classic step schedule; the measured raw sender of fabric_reduce_scatter is the reference.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"  // permanent stage zones (free when the profiler is off)
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_args.hpp"

using namespace tt::tt_fabric;
namespace mrs = matmul_reduce_scatter_prefill;

// One hop: 2D fabrics route by destination fabric node, 1D fabrics by hop count (ROUTING_MODE from the build).
inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    namespace ct = mrs::port_ct;
    namespace rt = mrs::port_sender_rt;
    namespace common = mrs::port_sender_common;
    constexpr uint32_t cb_port_chunks = get_compile_time_arg_val(ct::CB_PORT_CHUNKS);
    constexpr uint32_t P = get_compile_time_arg_val(ct::P);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(ct::CHUNK_TILES);
    constexpr uint32_t group = get_compile_time_arg_val(ct::GROUP);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(ct::TILE_BYTES);
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    constexpr uint32_t num_headers = group;
    constexpr uint32_t slot_bytes = P * tile_bytes;
    constexpr uint32_t num_waves = get_compile_time_arg_val(ct::NUM_WAVES);  // entry e is wave e mod num_waves
    // 1: every unit's last chunk per core carries the arrival increment; 0: only the block's last wave does (its
    // flush covers the earlier waves' chunks to the same core)
    constexpr bool inc_every_wave = get_compile_time_arg_val(ct::INC_EVERY_WAVE) != 0;
    constexpr uint32_t num_chunks = get_compile_time_arg_val(ct::NUM_CHUNKS);
    constexpr uint32_t zone_sets = get_compile_time_arg_val(ct::ZONE_SETS);

    // downstream zone of this direction (staging + offset)
    const uint32_t zone_base =
        get_common_arg_val<uint32_t>(common::STAGING_ADDR) + get_arg_val<uint32_t>(rt::ZONE_OFFSET);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(rt::ARRIVAL_ADDR);  // global arr_d on the downstream cores
    const uint32_t ready_addr = get_common_arg_val<uint32_t>(common::READY_ADDR);
    const uint32_t send_ready = get_arg_val<uint32_t>(rt::SEND_READY);
    const uint32_t ready_x = get_arg_val<uint32_t>(rt::READY_X);
    const uint32_t ready_y = get_arg_val<uint32_t>(rt::READY_Y);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(rt::DST_MESH_ID));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(rt::DST_CHIP_ID));
    const uint32_t num_cores = get_arg_val<uint32_t>(rt::NUM_CORES);
    const uint32_t num_units = get_arg_val<uint32_t>(rt::NUM_UNITS);
    // global credit counter (same address on every port)
    const uint32_t credit_addr = get_common_arg_val<uint32_t>(common::CREDIT_ADDR);
    const uint32_t wait_credits = get_arg_val<uint32_t>(rt::WAIT_CREDITS);  // credits this port waits for this call
    const uint32_t expect = get_arg_val<uint32_t>(rt::EXPECT);              // `freed` reports per consumed chunk
    const uint32_t fwd_total = get_arg_val<uint32_t>(rt::FWD_TOTAL);        // credits this port forwards this call
    const uint32_t ring_slots = get_arg_val<uint32_t>(rt::RING_SLOTS);      // downstream zone slots of this direction
    // downstream arrivals per chunk, half 0 / 1
    const uint32_t down_per_chunk0 = get_arg_val<uint32_t>(rt::DOWN_PER_CHUNK0);
    const uint32_t down_per_chunk1 = get_arg_val<uint32_t>(rt::DOWN_PER_CHUNK1);
    size_t arg = rt::COUNT;
    const uint32_t wave_tiles_idx = arg;  // per wave: tiles one core sends (<= P), dense at the front of its slot
    const uint32_t cores_idx = arg + num_waves;
    const uint32_t units_idx = cores_idx + num_cores;
    arg = units_idx + num_units;  // fabric connection args follow

    if (num_units == 0 && !send_ready && fwd_total == 0) {
        return;
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
    for (uint32_t h = 0; h < num_headers; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* credit_hdr = nullptr;
    if (fwd_total > 0) {
        credit_hdr = PacketHeaderPool::allocate_header();
        route_one_hop(credit_hdr, dst_chip_id, dst_mesh_id);
    }
    conn.open();

    // chunk credits (see header): forward one per chunk of the peer's arrivals my matmul cores consumed
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(mrs::sem::FREED));
    volatile tt_l1_ptr uint32_t* credits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_addr);
    uint32_t forwarded = 0;
    auto forward_poll = [&]() {
        if (forwarded >= fwd_total) {
            return;
        }
        invalidate_l1_cache();
        if (*freed >= (forwarded + 1) * expect) {
            credit_hdr->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, credit_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(credit_hdr), sizeof(PACKET_HEADER_TYPE));
            ++forwarded;
        }
    };

    // fence_ready
    if (send_ready) {
        hdrs[0]->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
    }
    if (num_units > 0) {
        DeviceZoneScopedN("READY_WAIT");
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        noc_semaphore_wait_min(ready, 1);
        noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
        noc_async_atomic_barrier();
    }

    // send_unit, mirroring the reader's walk and batch boundaries
    uint32_t rpos = 0, u = 0, lim = group, h = 0;
    auto pop_batch = [&]() {
        noc_async_writes_flushed();  // sources + headers of the in-flight packets have left L1
        cb_pop_front(cb_port_chunks, u * chunk_tiles);
        rpos += u;
        u = 0;
        lim = group - rpos % group;
    };
    for (uint32_t ug = 0; ug < num_chunks * num_units; ++ug) {
        DeviceZoneScopedN("SEND_UNIT");
        const uint32_t un = ug % num_units;
        const uint32_t chunk = ug / num_units;
        if (un == 0 && chunk >= zone_sets) {
            DeviceZoneScopedN("CREDIT_WAIT");
            invalidate_l1_cache();
            while (*credits < chunk - zone_sets + 1) {  // the downstream consumed chunk - zone_sets
                forward_poll();
                invalidate_l1_cache();
            }
        }
        forward_poll();
        const uint32_t unit = get_arg_val<uint32_t>(units_idx + un);
        const uint32_t half_mask = (unit >> 8) & 0x3;
        const uint32_t slot0 = (unit >> 16) & 0xFF;
        const uint32_t slot1 = (unit >> 24) & 0xFF;
        const uint32_t first_half = (unit >> 10) & 0x1;
        const uint32_t tiles = get_arg_val<uint32_t>(wave_tiles_idx + (unit & 0xFF) % num_waves);
        const uint32_t chunks_per_core = (tiles + chunk_tiles - 1) / chunk_tiles;
        const bool carries_inc = inc_every_wave || (unit & 0xFF) % num_waves == num_waves - 1;
        for (uint32_t walk = 0; walk < 2 * num_cores; ++walk) {  // the reader's walk: first_half's cores first
            const uint32_t hh = first_half ^ (walk >= num_cores ? 1u : 0u);
            const uint32_t i = walk >= num_cores ? walk - num_cores : walk;
            const uint32_t cw = get_arg_val<uint32_t>(cores_idx + i);
            const uint32_t hf = (cw >> 16) & 0xF;
            if (!((half_mask >> hh) & 1) || hf != hh) {
                continue;
            }
            const uint32_t x = cw & 0xFF, y = (cw >> 8) & 0xFF;
            // the downstream core's arrival ordinal (its half, this direction) on the zone ring
            const uint32_t ordinal = chunk * (hf ? down_per_chunk1 : down_per_chunk0) + (hf ? slot1 : slot0);
            const uint32_t dst_base = zone_base + (ordinal % ring_slots) * slot_bytes;
            const uint64_t ctr = get_noc_addr(x, y, arrival_addr);
            for (uint32_t ci = 0; ci < chunks_per_core; ++ci) {
                cb_wait_front(cb_port_chunks, chunk_tiles * (u + 1));  // per-chunk: no zone (marker budget)
                const uint32_t src = get_read_ptr(cb_port_chunks) + u * chunk_bytes;
                const uint32_t n = (tiles - ci * chunk_tiles) < chunk_tiles ? (tiles - ci * chunk_tiles) : chunk_tiles;
                const uint32_t bytes = n * tile_bytes;
                const uint64_t dst = get_noc_addr(x, y, dst_base + ci * chunk_bytes);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                if (carries_inc && ci + 1 == chunks_per_core) {  // a core's last chunk carries its arrival increment
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                if (++u == lim) {
                    pop_batch();
                }
            }
        }
        if (u > 0) {
            pop_batch();  // end of unit (the reader flushed there too)
        }
    }
    while (forwarded < fwd_total) {
        forward_poll();
    }
    conn.close();
    if (wait_credits > 0) {  // re-arm: every credit of this call has landed (the last chunk's was waited for)
        noc_semaphore_wait_min(credits, wait_credits);
        noc_semaphore_inc(get_noc_addr(credit_addr), 0u - wait_credits);
        noc_async_atomic_barrier();
    }
}
