// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port cores — shared compile-time layout and the chunk walk of one slot (TRANSPORT).
// Must match _port_ct() / _walk_chunks() in all_gather_rms_norm_program_descriptor.py.
//
// A slot is one chip's normalized slice at its FINAL destination pages: page id = q * slot_stride + rr * row_stride
// + jj for rr < rows_eff, jj < cols_eff (aligned W: the slice's columns inside the gathered output rows; R_U: the
// slot-contiguous staging scratch, one flat row). A chunk is a run of <= run_pages pages consecutive in ONE bank of
// the destination (page ids P, P + nb, P + 2 nb, ...), so one packet writes it with one NoC write; the same run is
// contiguous in the local copy a relay reads. Link l owns banks l, l + L, ...; part 1 / 2 = the first / second half of
// those banks (the far slot of a balanced ring). The walk is bank-major, so part 1 is a prefix of the part-0 walk.
#pragma once

#include <cstdint>

#include "agrn_common.hpp"

namespace agrn_port {

constexpr uint32_t cb_chunks = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t run_pages = get_compile_time_arg_val(2);
constexpr uint32_t group = get_compile_time_arg_val(3);      // chunks per batch (the port CB holds 2 x group)
constexpr uint32_t inc_every = get_compile_time_arg_val(4);  // arrival increment every N chunks (+ each entry's last)
constexpr uint32_t nb = get_compile_time_arg_val(5);         // banks of the destination buffer
constexpr uint32_t rows_eff = get_compile_time_arg_val(6);
constexpr uint32_t cols_eff = get_compile_time_arg_val(7);
constexpr uint32_t row_stride = get_compile_time_arg_val(8);
constexpr uint32_t slot_stride = get_compile_time_arg_val(9);
constexpr bool flat_slot = get_compile_time_arg_val(10) != 0;  // R_U: one flat row of R * Wlt pages
constexpr uint32_t Wlt = get_compile_time_arg_val(11);
constexpr uint32_t K = get_compile_time_arg_val(12);
constexpr uint32_t n_groups = get_compile_time_arg_val(13);
constexpr uint32_t rows_hi = get_compile_time_arg_val(14);
constexpr uint32_t big = get_compile_time_arg_val(15);
constexpr uint32_t cw_nom = get_compile_time_arg_val(16);
constexpr uint32_t payload_bytes = get_compile_time_arg_val(17);  // router max payload (stats packets)
constexpr uint32_t scatter_runs = get_compile_time_arg_val(18);   // runs per bulk packet (1..4, scatter write)
// > 0: the destination is an L1-sharded output with shard_w-tile-wide shards. Lanes are then tile-rows (link l owns
// rows l, l + L, ...) and a run is a slot row's columns inside one shard core (consecutive pages in its L1).
constexpr uint32_t shard_w = get_compile_time_arg_val(19);
constexpr uint32_t last_crt = get_compile_time_arg_val(20);  // tile-rows of the last row-group (own-entry pull)
// reader pipeline: packets whose reads are in flight at once (one NoC transaction id each, 1..15); each packet is
// pushed to the sender as soon as its own reads landed
constexpr uint32_t read_depth = get_compile_time_arg_val(21);
// own entry source: 0 = pull the pages from the workers' cb_out_block (L1), 1 = read the bank runs back from this
// chip's own destination pages (one read per run instead of one per page; the workers wrote them before own_ready)
constexpr uint32_t own_from_dst_ct = get_compile_time_arg_val(22);
// per chunk slot, the packet's runs (PacketDesc): written by the reader with the slot's reads, read by the sender once
// the slot is pushed, so only the reader walks the slot (the sender is bound by per-packet work)
constexpr uint32_t cb_desc = get_compile_time_arg_val(23);
// 1: a run that does not fit the open packet is split (its head fills the packet, the rest opens the next), so packets
// carry run_pages pages unless the scatter_runs limit closes them; 0: a run that does not fit closes the packet
constexpr bool split_runs = get_compile_time_arg_val(24) != 0;
// own-ready bands (OVERLAP): > 1 — the own entry waits, per packet, only for the bands (consecutive row-groups) its
// runs' tile-rows fall in; 1 — once for every active worker (sem_own_ready)
constexpr uint32_t own_bands = get_compile_time_arg_val(25);
// 1: after issuing a packet's reads, push every in-flight packet whose reads already landed (non-blocking trid
// check); 0: push only when read_depth packets are in flight (or before a blocking wait)
constexpr bool eager_push = get_compile_time_arg_val(26) != 0;
// EXPERIMENT own_entry_port_l1_staging: the workers write the own entry's leading pages into cb_stage (walk order)
constexpr uint32_t cb_stage = get_compile_time_arg_val(27);
constexpr bool stage_on = get_compile_time_arg_val(28) != 0;
constexpr uint32_t PORT_CT_ARGS = 29;  // TensorAccessorArgs(dst) follow
static_assert(own_bands >= 1 && own_bands <= 32, "band mask");
constexpr uint32_t run_stride = shard_w ? 1 : nb;        // page-id step between the pages of one run
constexpr uint32_t shard_w_div = shard_w ? shard_w : 1;  // (the interleaved walk never divides by it)
constexpr uint32_t chunk_bytes = run_pages * page_bytes;

static_assert(scatter_runs >= 1 && scatter_runs <= 4, "NOC_SCATTER_WRITE_MAX_CHUNKS = 4");
static_assert(
    read_depth >= 1 && read_depth <= 15 && read_depth <= group, "one trid per packet; the sender pops groups");

// One bulk packet: up to scatter_runs runs (each consecutive in one destination bank), <= run_pages pages in total,
// packed back to back in the port CB slot. run i = pages first[i], first[i] + nb, ... (n[i] of them).
struct Packet {
    uint32_t nr;
    uint32_t pages;
    uint32_t first[4];
    uint32_t n[4];
    uint32_t rr[4];
    uint32_t jj[4];
};

// One packet's runs as handed from the port reader to the sender (one cb_desc page per chunk slot).
struct PacketDesc {
    uint32_t nr;
    uint32_t pages;
    uint32_t first[4];
    uint32_t n[4];
    uint32_t src;  // EXPERIMENT: != 0 — send from this L1 address (staged own packet), else from the chunk slot
};
static_assert(sizeof(PacketDesc) <= 64, "cb_desc page");
constexpr uint32_t DESC_BYTES = 64;

// f(const Packet&): the packets of slot q for the lane (first_bank, bank_stride), part 0 / 1 / 2. Runs are visited
// bank-major; a packet closes when full (scatter_runs runs or run_pages pages) and always at the half-lane bank
// boundary, so the part-1 walk is a prefix of the part-0 walk and part 2 a suffix (balanced-ring relays).
template <typename F>
inline void for_each_packet(uint32_t q, uint32_t part, uint32_t first_bank, uint32_t bank_stride, F&& f) {
    const uint32_t n_lanes = shard_w ? rows_eff : nb;
    if (first_bank >= n_lanes) {
        return;
    }
    const uint32_t nl = (n_lanes - first_bank + bank_stride - 1) / bank_stride;
    const uint32_t lo = part == 2 ? nl / 2 : 0;
    const uint32_t hi = part == 1 ? nl / 2 : nl;
    const uint32_t base = q * slot_stride;
    Packet pk;
    pk.nr = 0;
    pk.pages = 0;
    auto flush = [&]() {
        if (pk.nr > 0) {
            f(pk);
            pk.nr = 0;
            pk.pages = 0;
        }
    };
    // one run of n pages (first, first + run_stride, ...): appended to the open packet, split over packets if allowed
    auto add_run = [&](uint32_t first, uint32_t n, uint32_t rr, uint32_t jj) {
        if constexpr (!split_runs) {
            if (pk.nr == scatter_runs || pk.pages + n > run_pages) {
                flush();
            }
            pk.first[pk.nr] = first;
            pk.n[pk.nr] = n;
            pk.rr[pk.nr] = rr;
            pk.jj[pk.nr] = jj;
            ++pk.nr;
            pk.pages += n;
            return;
        }
        while (n > 0) {
            if (pk.nr == scatter_runs || pk.pages == run_pages) {
                flush();
            }
            const uint32_t room = run_pages - pk.pages;
            const uint32_t m = n < room ? n : room;
            pk.first[pk.nr] = first;
            pk.n[pk.nr] = m;
            pk.rr[pk.nr] = rr;
            pk.jj[pk.nr] = jj;
            ++pk.nr;
            pk.pages += m;
            first += m * run_stride;
            jj += m * run_stride;
            n -= m;
        }
    };
    if constexpr (shard_w > 0) {
        // lane = tile-rows; a run never crosses a shard-column boundary of the destination
        for (uint32_t i = lo; i < hi; ++i) {
            if (i == nl / 2) {
                flush();
            }
            const uint32_t rr = first_bank + i * bank_stride;
            const uint32_t rb = base + rr * row_stride;
            uint32_t jj = 0;
            while (jj < cols_eff) {
                const uint32_t gc = base + jj;
                uint32_t seg_end = (gc / shard_w_div + 1) * shard_w_div - base;
                seg_end = seg_end < cols_eff ? seg_end : cols_eff;
                const uint32_t n = (seg_end - jj) < run_pages ? seg_end - jj : run_pages;
                add_run(rb + jj, n, rr, jj);
                jj += n;
            }
        }
        flush();
        return;
    }
    for (uint32_t i = lo; i < hi; ++i) {
        if (i == nl / 2) {
            flush();
        }
        const uint32_t b = first_bank + i * bank_stride;
        for (uint32_t rr = 0; rr < rows_eff; ++rr) {
            const uint32_t rb = base + rr * row_stride;
            uint32_t jj = (b + nb - rb % nb) % nb;  // first column of this row in bank b
            while (jj < cols_eff) {
                const uint32_t avail = (cols_eff - 1 - jj) / nb + 1;
                const uint32_t n = avail < run_pages ? avail : run_pages;
                add_run(rb + jj, n, rr, jj);
                jj += n * nb;
            }
        }
    }
    flush();
}

}  // namespace agrn_port
