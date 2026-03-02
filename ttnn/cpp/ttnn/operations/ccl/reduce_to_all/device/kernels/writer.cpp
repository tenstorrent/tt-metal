// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// FORWARDER APPROACH:
// Instead of using mux with connect/disconnect cycles, workers write
// complete packets to forwarder slots via NoC and increment forwarder semaphore.
// The forwarder kernel (running on forwarder cores) collects packets and forwards
// them via fabric.
//
// TYPE A/B WORKER SPLIT:
// Workers are classified as Type A or Type B based on (device_id + core_index) % 2:
//   - Type A: R1 sends via FWD forwarder, R2 sends via BWD forwarder
//   - Type B: R1 sends via BWD forwarder, R2 sends via FWD forwarder
// This balances FWD/BWD traffic in each round.
//
// TWO TRANSFER MODES (selected at compile time via single_shot_l flag):
//
// Single-shot mode (single_shot_l = 1):
//   When the full L tensor fits in one fabric packet (total_l_bytes <= max_fabric_payload).
//   Each worker sends 2 packets per round:
//   - MS packet (slot 0) → written to buffer END (offset total_l_bytes)
//   - Full L packet (slot 1) → written contiguously from offset 0
//
// Chunked streaming mode (single_shot_l = 0):
//   When L is too large for one packet, it is split into num_l_chunks chunks.
//   Each worker sends 1 + num_l_chunks packets per round:
//   - MS packet (slot 0) → written to buffer END (offset total_l_bytes)
//   - L chunks (slots 1..num_l_chunks) → written contiguously from offset 0
//   R2 uses streaming: sends each L chunk as soon as compute produces it.
//
// Buffer layout on receiver (both modes): [L_data][MS]
// L CB is aliased at buffer base (offset 0) for zero-copy receive.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#include <cstdint>

// =============================================================================
// Compile-time arguments
// =============================================================================
static constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
static constexpr uint32_t cb_local_ms = get_compile_time_arg_val(1);
static constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(2);
static constexpr uint32_t cb_r1_result_ms = get_compile_time_arg_val(3);
static constexpr uint32_t cb_packet_slot = get_compile_time_arg_val(4);
static constexpr uint32_t l1_alignment = get_compile_time_arg_val(5);
static constexpr uint32_t page_size_bytes = get_compile_time_arg_val(6);
static constexpr uint32_t slot_size = get_compile_time_arg_val(7);
static constexpr uint32_t ms_tile_size_bytes = get_compile_time_arg_val(8);
static constexpr uint32_t single_shot_l = get_compile_time_arg_val(9);   // 0 = chunked, 1 = single-shot
static constexpr uint32_t total_l_bytes = get_compile_time_arg_val(10);  // Full L size (always valid)
static constexpr uint32_t out_tiles = get_compile_time_arg_val(11);      // Total L tiles (always valid)
// Chunked mode parameters (only meaningful when single_shot_l == 0)
static constexpr uint32_t l_chunk_size_bytes = get_compile_time_arg_val(12);
static constexpr uint32_t num_l_chunks = get_compile_time_arg_val(13);
static constexpr uint32_t tiles_per_l_chunk = get_compile_time_arg_val(14);

static constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

// Slot indices: MS = slot 0, L = slot 1 (single-shot) or slots 1..num_l_chunks (chunked)
static constexpr uint32_t MS_SLOT_OFFSET = 0;
static constexpr uint32_t L_SLOT_OFFSET = 1;

// =============================================================================
// Address helper structs
// =============================================================================

/** Fabric destination addresses (where data lands on remote device) */
struct FabricDest {
    uint64_t dst_noc;  // Destination L1 address (varies per packet)
    uint64_t sem_noc;  // Semaphore address (constant per round)
};

/** Forwarder destination addresses (local forwarder slot) */
struct ForwarderDest {
    uint64_t slot_noc;  // Forwarder slot NOC address
    uint64_t sem_noc;   // Forwarder semaphore NOC address (constant per round)
    uint32_t slot_idx;  // Slot index for bit-packed signaling
};

// =============================================================================
// Configuration structs
// =============================================================================

/**
 * Per-round configuration for sending packets.
 * Groups destination info and forwarder slot info.
 */
struct RoundConfig {
    uint32_t cb_l;
    uint32_t cb_ms;
    uint32_t dst_mesh_id;
    uint32_t dst_chip_id;
    uint32_t dst_base_addr;
    uint32_t sem_addr;
    uint32_t fwd_slot_addr;
    uint32_t fwd_sem_addr;
    uint32_t base_slot_idx;
};

// =============================================================================
// PacketSender - unified sender supporting both single-shot and chunked modes
// =============================================================================

/**
 * Packet sender with cached core coordinates and round configuration.
 * Supports two transfer modes selected at compile time:
 * - Single-shot: sends full L tensor as one fabric packet
 * - Chunked: sends L as num_l_chunks separate packets, streaming in R2
 *
 * Template parameters encode size constants for zero-overhead abstraction.
 */
template <
    uint32_t slot_size,
    uint32_t ms_tile_size_bytes,
    uint32_t total_l_bytes,
    uint32_t out_tiles,
    uint32_t single_shot_l,
    uint32_t l_chunk_size_bytes,
    uint32_t num_l_chunks,
    uint32_t tiles_per_l_chunk>
struct PacketSender {
    // Core coordinates (constant across rounds)
    uint32_t current_core_x;
    uint32_t current_core_y;
    uint32_t fwd_core_x;
    uint32_t fwd_core_y;

    // Round configuration (set by setup_round)
    RoundConfig cfg;

    // Precomputed NOC addresses (computed once per round in setup_round)
    uint64_t sem_noc;      // Fabric destination semaphore
    uint64_t fwd_sem_noc;  // Forwarder semaphore

    // Cached header address (set once per round, reused for all packets)
    uint32_t header_addr;

    // =========================================================================
    // Round setup and teardown
    // =========================================================================

    /**
     * Configure sender for a new round.
     * - Copies config and precomputes constant NOC addresses
     * - Reserves header slot once for entire round
     * - Sets unicast route once (constant for all packets in round)
     */
    FORCE_INLINE void setup_round(const RoundConfig& new_cfg) {
        cfg = new_cfg;
        sem_noc = get_noc_addr(current_core_x, current_core_y, cfg.sem_addr);
        fwd_sem_noc = get_noc_addr(fwd_core_x, fwd_core_y, cfg.fwd_sem_addr);

        // Reserve header slot once for entire round (reused for all packets)
        cb_reserve_back(cb_packet_slot, 1);
        header_addr = get_write_ptr(cb_packet_slot);

        // Set unicast route once (constant for all packets in this round)
        auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
        (void)fabric_set_unicast_route(header, cfg.dst_chip_id, cfg.dst_mesh_id);
    }

    /**
     * Release header slot at end of round.
     * Must be called after all packets are sent for the round.
     */
    FORCE_INLINE void finish_round() {
        cb_push_back(cb_packet_slot, 1);
        cb_pop_front(cb_packet_slot, 1);
    }

    // =========================================================================
    // Address helpers
    // =========================================================================

    /**
     * Compute fabric destination addresses for a packet.
     * sem_noc is precomputed and constant per round.
     */
    FORCE_INLINE FabricDest get_fabric_dest(uint32_t dst_addr) const {
        return {
            get_noc_addr(current_core_x, current_core_y, dst_addr),
            sem_noc  // precomputed
        };
    }

    /**
     * Compute forwarder destination addresses for a packet.
     * fwd_sem_noc is precomputed and constant per round.
     *
     * @tparam is_ms True for MS packet (slot 0), false for L packet
     * @param l_chunk_idx L chunk index (only used in chunked mode for is_ms=false)
     */
    template <bool is_ms>
    FORCE_INLINE ForwarderDest get_forwarder_dest(uint32_t l_chunk_idx = 0) const {
        uint32_t slot_offset;
        if constexpr (is_ms) {
            slot_offset = MS_SLOT_OFFSET;
        } else if constexpr (single_shot_l) {
            slot_offset = L_SLOT_OFFSET;
        } else {
            slot_offset = L_SLOT_OFFSET + l_chunk_idx;
        }
        uint32_t slot_idx = cfg.base_slot_idx + slot_offset;
        uint32_t fwd_slot_addr = cfg.fwd_slot_addr + slot_offset * slot_size;
        return {
            get_noc_addr(fwd_core_x, fwd_core_y, fwd_slot_addr),
            fwd_sem_noc,  // precomputed
            slot_idx};
    }

    // =========================================================================
    // Generic packet send
    // =========================================================================

    /**
     * Generic packet send: update header command, then transfer to forwarder.
     *
     * @param fabric_dest Fabric destination addresses
     * @param fwd_dest Forwarder destination addresses
     * @param src_addr Source data address in L1
     * @param payload_size Payload size in bytes
     */
    FORCE_INLINE void send_packet(
        const FabricDest& fabric_dest, const ForwarderDest& fwd_dest, uint32_t src_addr, uint32_t payload_size) const {
        // Update header command (route already set in setup_round)
        auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
        constexpr uint32_t ATOMIC_INC_VAL = 1;
        constexpr bool FLUSH_WRITES = false;
        header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                fabric_dest.dst_noc, fabric_dest.sem_noc, ATOMIC_INC_VAL, FLUSH_WRITES},
            align(payload_size, l1_alignment));

        // Write header to forwarder slot
        noc_async_write(header_addr, fwd_dest.slot_noc, packet_header_size_bytes);

        // Write payload directly from source to forwarder slot
        uint64_t fwd_payload_noc = fwd_dest.slot_noc + packet_header_size_bytes;
        noc_async_write(src_addr, fwd_payload_noc, payload_size);

        // Wait for both writes to flush, then signal forwarder
        noc_async_writes_flushed();
        noc_semaphore_inc(fwd_dest.sem_noc, 1u << fwd_dest.slot_idx);
    }

    // =========================================================================
    // High-level send methods
    // =========================================================================

    /**
     * Send MS packet (slot 0).
     * MS is written to the END of the receive buffer (offset = total_l_bytes).
     */
    FORCE_INLINE void send_ms() const {
        uint32_t dst_addr = cfg.dst_base_addr + total_l_bytes;
        auto fabric_dest = get_fabric_dest(dst_addr);
        auto fwd_dest = get_forwarder_dest<true>();
        send_packet(fabric_dest, fwd_dest, get_read_ptr(cfg.cb_ms), ms_tile_size_bytes);
    }

    /**
     * Send full L tensor as a single packet (slot 1).
     * Only used in single-shot mode.
     */
    FORCE_INLINE void send_full_l() const {
        uint32_t dst_addr = cfg.dst_base_addr;
        uint32_t src_addr = get_read_ptr(cfg.cb_l);
        auto fabric_dest = get_fabric_dest(dst_addr);
        auto fwd_dest = get_forwarder_dest<false>();
        send_packet(fabric_dest, fwd_dest, src_addr, total_l_bytes);
    }

    /**
     * Send one L chunk packet (slots 1..num_l_chunks).
     * Only used in chunked streaming mode.
     *
     * @param l_chunk_idx L chunk index (0 to num_l_chunks-1)
     */
    FORCE_INLINE void send_l_chunk(uint32_t l_chunk_idx) const {
        uint32_t dst_addr = cfg.dst_base_addr + l_chunk_idx * l_chunk_size_bytes;
        uint32_t src_addr = get_read_ptr(cfg.cb_l) + l_chunk_idx * l_chunk_size_bytes;
        auto fabric_dest = get_fabric_dest(dst_addr);
        auto fwd_dest = get_forwarder_dest<false>(l_chunk_idx);
        send_packet(fabric_dest, fwd_dest, src_addr, l_chunk_size_bytes);
    }

    /**
     * Send all packets for a round (MS first, then L).
     * Used when all data is ready (R1 with local input).
     */
    FORCE_INLINE void send_all() const {
        send_ms();
        if constexpr (single_shot_l) {
            send_full_l();
        } else {
            for (uint32_t i = 0; i < num_l_chunks; i++) {
                send_l_chunk(i);
            }
        }
    }

    /**
     * Wait for compute output, then send all packets.
     * In single-shot mode: wait for MS, send MS, wait for all L, send full L.
     * In chunked mode: wait for MS, send MS, then stream L chunks as produced.
     */
    FORCE_INLINE void send_all_after_wait() const {
        cb_wait_front(cfg.cb_ms, 1);
        send_ms();
        if constexpr (single_shot_l) {
            cb_wait_front(cfg.cb_l, out_tiles);
            send_full_l();
        } else {
            // Stream L chunks: wait for each chunk as compute produces it
            for (uint32_t i = 0; i < num_l_chunks; i++) {
                // Cumulative wait: cb_wait_front waits for total tiles in CB
                cb_wait_front(cfg.cb_l, (i + 1) * tiles_per_l_chunk);
                send_l_chunk(i);
            }
        }
    }
};

// Type alias for convenience
using Sender = PacketSender<
    slot_size,
    ms_tile_size_bytes,
    total_l_bytes,
    out_tiles,
    single_shot_l,
    l_chunk_size_bytes,
    num_l_chunks,
    tiles_per_l_chunk>;

void kernel_main() {
    size_t arg_idx = 0;

    // R1 destination - intermediate tensor address on neighbor device
    const uint32_t r1_dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    // R2 destination - intermediate tensor address on neighbor device
    const uint32_t r2_dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    // Core coordinates
    const uint32_t current_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t current_core_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fwd_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fwd_core_y = get_arg_val<uint32_t>(arg_idx++);

    // R1 forwarder slot info
    const uint32_t r1_fwd_slot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t r1_base_slot_idx = get_arg_val<uint32_t>(arg_idx++);

    // R2 forwarder slot info
    const uint32_t r2_fwd_slot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t r2_base_slot_idx = get_arg_val<uint32_t>(arg_idx++);

    // Initialize sender with core coordinates
    Sender sender{current_core_x, current_core_y, fwd_core_x, fwd_core_y};

    // ==========================================================================
    // ROUND 1: Send local input to R1 neighbor via forwarder
    // All local data is ready, send all packets immediately
    // ==========================================================================
    {
        DeviceZoneScopedN("R1-SEND");
        sender.setup_round(
            {cb_local_l,
             cb_local_ms,
             r1_dst_mesh_id,
             r1_dst_chip_id,
             r1_neighbor_dst_addr,
             r1_neighbor_sem_addr,
             r1_fwd_slot_addr,
             r1_fwd_sem_addr,
             r1_base_slot_idx});
        sender.send_all();
        sender.finish_round();
    }

    // ==========================================================================
    // ROUND 2: Send R1 result to R2 neighbor via forwarder
    // Wait for R1 compute output, then send.
    // In chunked mode: streams L chunks as compute produces them.
    // In single-shot mode: waits for all L tiles then sends as one packet.
    // ==========================================================================
    {
        DeviceZoneScopedN("R2-SEND");
        sender.setup_round(
            {cb_r1_result_l,
             cb_r1_result_ms,
             r2_dst_mesh_id,
             r2_dst_chip_id,
             r2_neighbor_dst_addr,
             r2_neighbor_sem_addr,
             r2_fwd_slot_addr,
             r2_fwd_sem_addr,
             r2_base_slot_idx});
        sender.send_all_after_wait();
        sender.finish_round();
    }

    noc_async_full_barrier();
}
