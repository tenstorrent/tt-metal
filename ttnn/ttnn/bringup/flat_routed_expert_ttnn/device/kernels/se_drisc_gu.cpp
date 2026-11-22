// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat spatial expert, DRISC mode (se_drisc.hpp): the persistent gate/up weight sender of NJ reader slots (1 or 2:
// both of a bank's slots on one tile), on a DRAM core (launched once per device by flat_drisc.cpp, serves every flat
// call until the host sets its stop word).
//
// Per call each slot k posts a job (se_drisc_go.cpp): its chunks are its loads' (expert e: chunks e * PER_E .., each
// RG blocks of BLK bytes, from this bank's region). Unit u = (chunk c, block b) is read from GDDR by DMA (job k on
// stream k, which completes in order) into one of the job's NS L1 slots, then written to the G receivers that share
// block b (receiver j gets block j / G) at landing slot (region of c's load) * PER_E + c % PER_E, once each of them
// has granted chunk c, on the job's NoC with transaction id TRID0 + u % NS; when the id has no outstanding writes the
// receivers' data semaphores are bumped (in order: each receiver pushes its landing slots by count) and the slot
// refills. A receiver counts its grants in a semaphore of its own (zero at each launch); the sender reads the counters
// when it runs out of grants. The jobs advance round-robin.
//
// RT: 0 working base (se_drisc.hpp layout), 1 working region bytes, 2 NJ, 3.. NJ x the job's NoC, then the MPFE ports
// of
//     this tile and of the bank's NOC0 / NOC1 endpoints
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/gddr_dma.h"
#include "experimental/gddr_mc.h"
#include "se_drisc.hpp"

// the weight writes' unicast VC (as se10_fwd.cpp: off the default write VC the h / y traffic shares); SE_FWD_VC
#ifdef SE_FWD_VC
constexpr uint8_t fwd_vc = SE_FWD_VC;
#else
constexpr uint8_t fwd_vc = 3;
#endif

// (perf probe SE_DRISC_LEAN: fewer register accesses a unit. 1: the DMA's source-high register (always 0: bank-local
// sources) written once a call, not per DMA; the DMA completion count re-read only when the known completions are used
// up. 2: also no ready poll before a semaphore increment (unsafe in general: measures what those reads cost). 3 (the
// tensor prefetcher's scheme, on top of 1): POSTED block writes (no transaction id, no acks), the receivers' increments
// on the same static VC (FWD_VC) as the data, sent once the NIU's posted-write-sent counter has passed the unit's last
// write (same VC, same route: in order behind the data, the noc_async_writes_flushed + semaphore_inc pattern); one
// counter read a pass completes every unit that left, in order, and frees their slots (units in flight up to
// SE_DRISC_NS_MAX, at most 16 - no transaction ids bound it). 4: 3 without the ready poll before an increment
// (measure-only, as 2))
#ifndef SE_DRISC_LEAN
#define SE_DRISC_LEAN 0
#endif
#if SE_DRISC_LEAN >= 3 && defined(SE_DRISC_FREE_SENT)
#error "SE_DRISC_LEAN >= 3 frees slots once sent already: not with FREE_SENT"
#endif
#if SE_DRISC_LEAN >= 3
constexpr uint8_t inc_vc = fwd_vc;
constexpr uint32_t wr_resp = 0;  // (posted)
#else
constexpr uint8_t inc_vc = NOC_UNICAST_WRITE_VC;
constexpr uint32_t wr_resp = NOC_CMD_RESP_MARKED;
#endif

// The hot loop's NoC commands with their constant registers set once per call (every register access costs tens of
// cycles on a DRAM core: ~800 cycles per 9 KB block with the stock calls). Writes: the write command buffer keeps
// NOC_CTRL (VC FWD_VC), the length and the high address; per write only the source, destination, coordinate, id.
// Atomic increments of the receivers' data semaphore (one address): only the coordinate changes.
inline void fast_write_state(uint8_t noc, uint32_t bytes) {
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(fwd_vc) | wr_resp);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, bytes);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, 0);
#if SE_DRISC_LEAN >= 3
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_PACKET_TAG, 0);
#endif
}
inline void fast_write(uint8_t noc, uint32_t src, uint32_t coord, uint32_t dst, uint32_t trid) {
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
#if SE_DRISC_LEAN < 3
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
#endif
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
#if SE_DRISC_LEAN >= 3
    noc_posted_writes_num_issued[noc] += 1;
#else
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
#endif
}
#if SE_DRISC_LEAN >= 3
// (blocks over one NoC packet: posted packets of at most NOC_MAX_BURST_SIZE, the length set per packet)
inline void fast_write_long(uint8_t noc, uint32_t src, uint32_t coord, uint32_t dst, uint32_t bytes) {
    for (uint32_t o = 0; o < bytes; o += NOC_MAX_BURST_SIZE) {
        while (!noc_cmd_buf_ready(noc, write_cmd_buf));
        NOC_CMD_BUF_WRITE_REG(
            noc, write_cmd_buf, NOC_AT_LEN_BE, bytes - o < NOC_MAX_BURST_SIZE ? bytes - o : NOC_MAX_BURST_SIZE);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src + o);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dst + o);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, coord);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_posted_writes_num_issued[noc] += 1;
    }
}
#endif
inline void fast_inc_state(uint8_t noc, uint32_t addr) {
    while (!noc_cmd_buf_ready(noc, write_at_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_TARG_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_at_cmd_buf,
        NOC_CTRL,
        NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(inc_vc) | NOC_CMD_RESP_MARKED | NOC_CMD_AT);
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_at_cmd_buf,
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(31) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_AT_DATA, 1);
}
inline void fast_inc(uint8_t noc, uint32_t coord) {
    while (!noc_cmd_buf_ready(noc, write_at_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_TARG_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_CMD_CTRL, 0x1);
    noc_nonposted_atomics_acked[noc] += 1;
}

inline void lean_dma_read(uint8_t stream, uint32_t src, uint32_t dst, uint32_t attrs) {
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_READ_TRANSFER_SOURCE_LOW_REG_OFFSET, src);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_READ_TRANSFER_DEST_REG_OFFSET, dst);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs);
}
inline void fast_inc_nopoll(uint8_t noc, uint32_t coord) {
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_TARG_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_CMD_CTRL, 0x1);
    noc_nonposted_atomics_acked[noc] += 1;
}

// (perf probe SE_DRISC_XEMU: x's DRAM reads emulated inside the DRISC's queue - SE_DRISC_XEMU_B-byte reads from this
// bank into a scratch slot, never forwarded. Reads are owed by SE_DRISC_XEMU_MODE: 0 one every SE_DRISC_XEMU weight
// units (spread), 1 SE_DRISC_XEMU_K at each load's first unit, 2 the same one load ahead; issued with at most
// SE_DRISC_XEMU_CAP in flight, before (SE_DRISC_XEMU_FIRST) or after the pass's weight DMAs, on the job's stream or
// the other one (SE_DRISC_XEMU_OTHER))
#ifdef SE_DRISC_XEMU
#ifndef SE_DRISC_XEMU_B
#define SE_DRISC_XEMU_B 12288
#endif
#ifndef SE_DRISC_XEMU_MODE
#define SE_DRISC_XEMU_MODE 0
#endif
#ifndef SE_DRISC_XEMU_K
#define SE_DRISC_XEMU_K 5
#endif
#ifndef SE_DRISC_XEMU_CAP
#define SE_DRISC_XEMU_CAP 2
#endif
#ifndef SE_DRISC_XEMU_OTHER
#define SE_DRISC_XEMU_OTHER 0
#endif
#ifndef SE_DRISC_XEMU_FIRST
#define SE_DRISC_XEMU_FIRST 0
#endif
#ifndef SE_DRISC_XEMU_W  // 1: each completed x read also written out (stock NoC write, transaction id 15) to a receiver
#define SE_DRISC_XEMU_W 0  // core's landing ring - the DRISC -> relay delivery of x, emulated (garbage there)
#endif
static_assert(SE_DRISC_XEMU_CAP >= 1 && SE_DRISC_XEMU_CAP <= 8, "x emulation: 1..8 reads in flight (ring of 8)");
#define SE_XEMU_SAME (!SE_DRISC_XEMU_OTHER)
#ifdef SE_DRISC_X
#error "SE_DRISC_X and SE_DRISC_XEMU together"
#endif
#elif defined(SE_DRISC_X)
#define SE_XEMU_SAME 1  // (the x segment reads share the job's in-order stream: the same accounting)
#else
#define SE_XEMU_SAME 0
#endif

#ifdef SE_DRISC_X
// one x segment write from DRISC L1 into a relay's row-major chunk, transaction id 15 (every x write: one flush check
// covers them), on the register command buffer - the write buffer keeps the fast weight writes' constant registers
inline void x_write(uint8_t noc, uint32_t src, uint32_t coord, uint32_t dst, uint32_t bytes) {
    while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_reg_cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, bytes);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(15));
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_TARG_ADDR_LO, src);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
}
#endif

struct Job {
    uint32_t gddr, chunk, blk, rg, R, G, per_e, land, units, slot0, ns, trid0, ub, dmax;
    uint32_t rd, wr, fl, sn;  // units DMA'd, written, flushed (acked), sent (left this L1: SE_DRISC_FREE_SENT)
    uint32_t dn, dattr;       // (SE_DRISC_LEAN) units known DMA'd (in-order stream), the DMA's attribute word
#if SE_DRISC_LEAN >= 3
    uint32_t pm[16];  // (posted writes) the NoC's posted-write issue count after each in-flight unit's last write
#endif
    // (SE_DRISC_XEMU) DMAs issued on the job's stream (weights + extra), the extra reads' stream indices (ring of 8,
    // at most 2 in flight), extra reads completed, weight units since the last extra read, the scratch slot (0: off)
    uint32_t tot, xq[8], xh, xt, xdone, since, xslot, xpend, xi;
    uint8_t noc;
    uint8_t loads[SD_MAX_LOADS], regions[SD_MAX_LOADS];
    uint32_t cr[SE_DRISC_MAX_R];
    uint64_t rx[SE_DRISC_MAX_R], rdata[SE_DRISC_MAX_R], rgrant[SE_DRISC_MAX_R];
    uint32_t coord[SE_DRISC_MAX_R];  // (NOC_*_ADDR_COORDINATE of receiver j)
    uint32_t dsem;
#ifdef SE_DRISC_X
    // x from this DRISC (se_drisc.hpp XI_*): the call's info, the cursor over the relays' chunk stream (entry,
    // sub-block, super-block, row tile, row; the global super-block, its relay offset, the chunk's number for that
    // offset and its rows), the x tasks (in DMA order = the xq ring: chunk, row, relay offset), written / flushed +
    // counted, the relays' credit words as last read
    uint32_t xon, xia, xs0, x_seg, x_mt, x_nsb, x_vs, x_nrect, x_rmc, x_batch, x_page, x_xaddr, x_ring, x_rsva;
    uint32_t x_slot, x_banks, x_spb, x_nact, x_arra[XI_MAX_RMC];
    uint64_t x_tidx, x_rbase[XI_MAX_RL];
    uint32_t x_rcoord[XI_MAX_RL], x_rsv[XI_MAX_RL];
    uint32_t ca, cs, cj, cm, crow, cg, co, cqc, crows, coff, ccnt, cq[XI_MAX_RL];
    bool cin, cch, cdone;
    uint32_t tq[8], tr[8], to_[8], xw, xf;
#endif
};

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0), region = get_arg_val<uint32_t>(1), nj = get_arg_val<uint32_t>(2);
    auto word = [&](uint32_t off) { return reinterpret_cast<volatile uint32_t*>(base + off); };
    for (uint32_t o = 0; o < se_drisc_desc(0); o += 4) {
        if (o != SE_DRISC_STOP && (o < SE_DRISC_DONE + 4 || o > SE_DRISC_DONE + 12)) {
            *word(o) = 0;
        }
    }
    for (uint32_t k = 0; k < nj; ++k) {
        if (get_arg_val<uint32_t>(3 + k) != noc_index) {
            noc_local_state_init(get_arg_val<uint32_t>(3 + k));  // (the kernel start set up its own NoC only)
        }
    }
    for (uint32_t port = 1; port <= 3; ++port) {  // (debug: the controller's MPFE weights as found, P1..P3)
        *word(SE_DRISC_DONE + 4 * port) = gddr_mc_read_mpfe_weight(port);
    }
    for (uint32_t o = 496; o < 512; o += 4) {  // (debug words: the x emulation's state)
        *word(o) = 0;
    }
    *word(SE_DRISC_READY) = SE_DRISC_READY_MAGIC;
    // GDDR MPFE weights (the job descriptor's word, per call): this tile's port, the bank's NOC0 / NOC1 endpoints'
    // ports (RT 3 + NJ ..); the weights as found are restored when the call asks for none and on exit (they persist
    // until a reset)
    const uint32_t mp_port[3] = {
        get_arg_val<uint32_t>(3 + nj), get_arg_val<uint32_t>(4 + nj), get_arg_val<uint32_t>(5 + nj)};
    uint32_t mp_orig[3], mp_applied = 0;
    for (uint32_t q = 0; q < 3; ++q) {
        mp_orig[q] = gddr_mc_read_mpfe_weight(mp_port[q]);
    }
    auto mpfe = [&](uint32_t w) {
        for (uint32_t q = 0; q < 3; ++q) {
            gddr_mc_write_mpfe_weight(mp_port[q], w ? (w >> (4 * q)) & 7 : mp_orig[q]);
        }
        mp_applied = w;
    };
#ifndef SE_DRISC_BURST
#define SE_DRISC_BURST 255
#endif
    if constexpr (SE_DRISC_BURST != 0) {  // (0: leave the DMA transfer attributes alone)
        experimental::dma_set_burst_size(SE_DRISC_BURST);
    }
    uint32_t seen_real[2] = {0, 0}, seen_skip[2] = {0, 0};
    uint32_t seen = 0, dma_total[2] = {0, 0};  // (debug: this kernel's DMA reads per stream)
    const uint32_t hw0[2] = {
        READ_TX_STREAM_REG(0, TX_REG_STREAM_TOTAL_READS_REG_OFFSET),
        READ_TX_STREAM_REG(1, TX_REG_STREAM_TOTAL_READS_REG_OFFSET)};
    Job jobs[2];
    bool stop = false;
    while (!stop) {
        invalidate_l1_cache();
        if (*word(SE_DRISC_STOP)) {
            break;
        }
        // each job gets one post per call: a descriptor (job counter) or a skip (skip counter)
        bool posted = true;
        for (uint32_t k = 0; k < nj; ++k) {
            posted = posted && (*word(se_drisc_jobs(k)) != seen_real[k] || *word(se_drisc_skips(k)) != seen_skip[k]);
        }
        if (!posted) {
            continue;
        }
        ++seen;
        bool real[2] = {false, false};
        uint32_t n_real = 0;
        for (uint32_t k = 0; k < nj; ++k) {
            real[k] = *word(se_drisc_jobs(k)) != seen_real[k];
            if (real[k]) {
                ++seen_real[k];
                ++n_real;
            } else {
                ++seen_skip[k];
            }
        }
#ifdef SE_DRISC_X
        constexpr uint32_t x_res = 2 * SE_DRISC_XA;  // (the jobs' x areas, se_drisc.hpp)
#else
        constexpr uint32_t x_res = 0;
#endif
        const uint32_t room =
            n_real ? (region - SE_DRISC_SLOTS - x_res) / n_real : 0;  // (L1 slots split over the real jobs)
        uint32_t left = 0, ri = 0;
        for (uint32_t k = 0; k < nj; ++k) {
            Job& J = jobs[k];
            if (!real[k]) {
                J.units = 0;
                J.rd = J.wr = J.fl = J.sn = 0;
                continue;
            }
            volatile uint32_t* d = word(se_drisc_desc(k));
            J.gddr = d[SD_GDDR];
            J.chunk = d[SD_CHUNK];
            J.blk = d[SD_BLK];
            J.rg = d[SD_RG];
            J.R = d[SD_R];
            J.G = d[SD_G];
            J.per_e = d[SD_PER_E];
            J.land = d[SD_LAND];
            const uint32_t dsem = d[SD_DATA_SEM], n_load = d[SD_N_LOAD], gsem = d[SD_GRANTS];
            J.dsem = dsem;
#ifdef SE_DRISC_DMA_MAX  // (probe MIMO_FL_DRISC_DMA_MAX: DMA reads in flight)
            J.dmax = SE_DRISC_DMA_MAX;
#else
            J.dmax = d[SD_PAIRED] ? 2 : 255;  // (se_drisc.hpp SD_PAIRED)
#endif
            for (uint32_t l = 0; l < n_load; ++l) {
                const uint32_t v = d[SD_LOADS + l];
                J.loads[l] = v & 0xFF;
                J.regions[l] = (v >> 8) & 0xFF;
            }
            J.noc = get_arg_val<uint32_t>(3 + k);
            for (uint32_t j = 0; j < J.R; ++j) {
                const uint32_t xy = d[SD_RXY + j];
                J.rx[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, 0, J.noc);
                J.rdata[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, dsem, J.noc);
                J.rgrant[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, gsem, J.noc);
                J.coord[j] = (uint32_t)(J.rx[j] >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
                J.cr[j] = 0;
            }
            J.slot0 = base + SE_DRISC_SLOTS + (ri++) * room;
            // a unit is UB consecutive blocks of one chunk (one DMA, one transaction id, one flush check). One job
            // per tile: single blocks (the deepest pipeline: GLM-Flash M 64 33.9 vs 35.2 us / expert with pairs); two
            // jobs per tile (half the L1 each): pairs (the per-block loop, ~800 cycles, caps a tile at ~15 GB/s: all
            // 16 slots on one tile per bank M 64 57.8 -> 41.2 us). SE_DRISC_UB forces it.
#ifdef SE_DRISC_UB
            J.ub = SE_DRISC_UB;
#else
            J.ub = n_real > 1 && J.rg % 2 == 0 ? 2 : 1;
#endif
            J.ns = room / (J.ub * J.blk);
#ifndef SE_DRISC_NS_MAX
#define SE_DRISC_NS_MAX 7
#endif
            // (transaction ids 1-7 / 8-14 per job; probe SE_DRISC_NS_MAX up to 14: one real job takes ids 1-14, so
            // more units in flight when its L1 room holds more than 7)
            // (below 7 it lowers the cap: e.g. the x emulation's baseline with its slot count)
#if SE_DRISC_LEAN >= 3  // (posted: no transaction ids, only pm[] bounds the units in flight)
            const uint32_t ns_max = SE_DRISC_NS_MAX > 16 ? 16 : SE_DRISC_NS_MAX;
#else
            const uint32_t ns_max =
                SE_DRISC_NS_MAX < 7
                    ? SE_DRISC_NS_MAX
                    : (SE_DRISC_NS_MAX > 7 && n_real == 1 ? (SE_DRISC_NS_MAX > 14 ? 14 : SE_DRISC_NS_MAX) : 7);
#endif
            J.ns = J.ns > ns_max ? ns_max : J.ns;
            J.trid0 = ns_max > 7 ? 1 : 1 + 7 * k;
            J.units = n_load * J.per_e * (J.rg / J.ub);
            J.rd = J.wr = J.fl = J.sn = J.dn = 0;
            J.tot = J.xh = J.xt = J.xdone = J.since = J.xpend = J.xi = 0;
#ifdef SE_DRISC_XEMU
#if defined(SE_DRISC_FREE_SENT) || defined(SE_DRISC_DRAIN)
#error "SE_DRISC_XEMU: not with FREE_SENT / DRAIN"
#endif
#if SE_DRISC_LEAN >= 3
#error "SE_DRISC_XEMU: not with LEAN >= 3"
#endif
            // (perf probe: an extra SE_DRISC_XEMU_B-byte read from this bank every SE_DRISC_XEMU weight units, into a
            // scratch slot past the job's slots, never forwarded: x's DRAM reads moved into the DRISC's own queue)
            if (J.ns * J.ub * J.blk + SE_DRISC_XEMU_B > room) {  // (no room past the slots: one slot fewer)
                J.ns = (room - SE_DRISC_XEMU_B) / (J.ub * J.blk);
            }
            J.xslot = J.ns ? J.slot0 + J.ns * J.ub * J.blk : 0;
            *word(496) = J.ns;  // (debug words 496..511 of the working region, the dump's "xemu")
            *word(500) = J.xslot;
            *word(508) = room;
#else
            J.xslot = 0;
#endif
#ifdef SE_DRISC_X
            J.xon = d[SD_XON];
            J.xw = J.xf = 0;
            if (J.xon) {
                J.xia = base + region - (k + 1) * SE_DRISC_XA;
                volatile uint32_t* xi = reinterpret_cast<volatile uint32_t*>(J.xia);
                J.x_xaddr = xi[XI_XADDR];
                J.x_page = xi[XI_PAGE];
                J.x_mt = xi[XI_MT];
                J.x_nsb = xi[XI_NSB];
                J.x_vs = xi[XI_VSTRIDE];
                J.x_nrect = xi[XI_NRECT];
                J.x_rmc = xi[XI_RMC];
                J.x_batch = xi[XI_BATCH];
                J.x_seg = xi[XI_SEG];
                J.x_ring = xi[XI_RING];
                J.x_rsva = xi[XI_RSV];
                for (uint32_t p = 0; p < XI_MAX_RMC; ++p) {
                    J.x_arra[p] = xi[XI_ARR + p];
                }
                J.x_slot = xi[XI_SLOT];
                J.x_banks = xi[XI_BANKS];
                J.x_spb = xi[XI_SPB];
                J.x_nact = xi[XI_NACT];
                const uint32_t tw = J.noc ? XI_TIDX1 : XI_TIDX0;
                J.x_tidx = uint64_t(xi[tw]) | (uint64_t(xi[tw + 1]) << 32);
                for (uint32_t i = 0; i < J.x_vs * J.x_nrect; ++i) {
                    const uint32_t xy = xi[XI_RXY + i];
                    J.x_rbase[i] = get_noc_addr(xy >> 16, xy & 0xFFFF, 0, J.noc);
                    J.x_rcoord[i] = (uint32_t)(J.x_rbase[i] >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
                    J.x_rsv[i] = 0;
                }
                for (uint32_t o = 0; o < XI_MAX_RL; ++o) {
                    J.cq[o] = 0;
                }
                J.ca = J.cs = J.cj = J.cm = J.crow = J.cg = 0;
                J.cin = J.cch = J.cdone = false;
                // the x segment slots out of the job's room (the weight slots shrink to fit)
                const uint32_t xs_bytes = SE_DRISC_XS * J.x_seg;
                if (J.ns * J.ub * J.blk + xs_bytes > room) {
                    J.ns = (room - xs_bytes) / (J.ub * J.blk);
                }
                J.xs0 = J.slot0 + J.ns * J.ub * J.blk;
            }
#endif
            left += J.units;
            volatile uint32_t* st = word(se_drisc_state(k));
            st[0] = seen;
            st[1] = J.units;
            st[5] = n_load;
        }
        for (uint32_t k = 0; k < nj; ++k) {  // (the MPFE weights of this call: the first real job's descriptor)
            if (real[k]) {
                if (const uint32_t w = word(se_drisc_desc(k))[SD_MPFE]; w != mp_applied) {
                    mpfe(w);
                }
                break;
            }
        }
#if SE_DRISC_LEAN
#ifndef SE_DRISC_STREAM0
#define SE_DRISC_STREAM0 0
#endif
        for (uint32_t k = 0; k < nj; ++k) {
            if (jobs[k].units) {
                const uint8_t s = (SE_DRISC_STREAM0 + k) & 1;
                WRITE_TX_STREAM_REG(s, TX_REG_STREAM_READ_TRANSFER_SOURCE_HIGH_REG_OFFSET, 0);
                DmaTxqTransferAttrs_u a = {.val = DmaTxqTransferAttrs_DEFAULT};
                a.f.transfer_size_words = (jobs[k].ub * jobs[k].blk) >> 4;
                a.f.transfer_start_read = 1;
                jobs[k].dattr = a.val;
            }
        }
#endif
        for (uint32_t k = 0; k < nj; ++k) {  // (the fast path's constant registers, per NoC)
#if SE_DRISC_LEAN >= 3
            if (jobs[k].units) {  // (posted: every write and increment on the fast path; nothing posted is in flight
                                  // between calls, so the issue count restarts from the NIU's sent count)
                const uint8_t n = jobs[k].noc;
                noc_posted_writes_num_issued[n] = NOC_STATUS_READ_REG(n, NIU_MST_POSTED_WR_REQ_SENT);
                fast_write_state(n, jobs[k].blk <= NOC_MAX_BURST_SIZE ? jobs[k].blk : NOC_MAX_BURST_SIZE);
                fast_inc_state(n, jobs[k].dsem);
            }
#else
            if (jobs[k].units && jobs[k].blk <= NOC_MAX_BURST_SIZE) {
                fast_write_state(jobs[k].noc, jobs[k].blk);
                fast_inc_state(jobs[k].noc, jobs[k].dsem);
            }
#endif
        }
#if SE_XEMU_SAME
        // (x reads on the job's in-order stream: completed DMAs less the x reads among them = weight units done)
        auto xrefresh = [&](Job& J, uint8_t s) {
            const uint32_t d = J.tot - experimental::dma_get_reads_outstanding(s);
            while (J.xh != J.xt && J.xq[J.xh & 7] < d) {
                ++J.xh;
                ++J.xdone;
#if SE_DRISC_XEMU_W
                // (on the read command buffer: the write one keeps the fast weight writes' constant registers; the
                // grant refreshes' stock reads reprogram this one every time)
                noc_async_write_one_packet_with_trid(
                    J.xslot, J.rx[J.xdone % J.R] | J.land, SE_DRISC_XEMU_B, 15, read_cmd_buf, J.noc, fwd_vc);
#endif
            }
            J.dn = d - J.xdone;
        };
#endif
#ifdef SE_DRISC_X
        // The relays' chunk stream, as se11_xrd.cpp walks it: entries a (off, cnt), their sub-blocks of MT x 32 rows,
        // super-blocks j (the global count g picks the relay offset g % VSTRIDE; every rectangle's relay at that
        // offset takes the chunk), row tiles m; chunk q counts per offset. The next row segment this DRISC owns (its
        // bank's rows, its share of them): token t, super-block j, offset o, chunk q, row r. A sub-block's token ids
        // are read once (NoC, from the token index row).
        auto x_next = [&](Job& J, uint32_t& t, uint32_t& j, uint32_t& o, uint32_t& q, uint32_t& r) {
            volatile uint32_t* xi = reinterpret_cast<volatile uint32_t*>(J.xia);
            volatile uint32_t* idx = reinterpret_cast<volatile uint32_t*>(J.xia + XI_IDX_OFF);
            const uint32_t rps = J.x_mt * 32;
            while (true) {
                if (J.ca >= J.x_nact) {
                    return false;
                }
                if (!J.cin) {
                    J.coff = xi[XI_ENT + 2 * J.ca];
                    J.ccnt = xi[XI_ENT + 2 * J.ca + 1];
                    if (J.cs * rps >= J.ccnt) {
                        ++J.ca;
                        J.cs = 0;
                        continue;
                    }
                    const uint32_t rows = J.ccnt - J.cs * rps < rps ? J.ccnt - J.cs * rps : rps;
                    noc_async_read(
                        J.x_tidx + (J.coff + J.cs * rps) * 4, J.xia + XI_IDX_OFF, (rows + 31) / 32 * 128, J.noc);
                    noc_async_read_barrier(J.noc);
                    invalidate_l1_cache();
                    J.cin = true;
                    J.cch = false;
                    J.cj = J.cm = 0;
                }
                if (J.cj >= J.x_nsb) {
                    J.cin = false;
                    ++J.cs;
                    continue;
                }
                if (!J.cch) {
                    J.co = J.cg % J.x_vs;
                    J.cqc = J.cq[J.co]++;
                    const uint32_t r0 = J.cs * rps + J.cm * 32;
                    J.crows = r0 < J.ccnt ? (J.ccnt - r0 < 32 ? J.ccnt - r0 : 32) : 0;
                    J.crow = 0;
                    J.cch = true;
                }
                while (J.crow < J.crows) {
                    const uint32_t rr = J.crow++, tok = idx[J.cm * 32 + rr];
                    if (tok % J.x_banks == J.x_slot % J.x_banks &&
                        (tok / J.x_banks) % J.x_spb == J.x_slot / J.x_banks) {
                        t = tok;
                        j = J.cj;
                        o = J.co;
                        q = J.cqc;
                        r = rr;
                        return true;
                    }
                }
                J.cch = false;
                if (++J.cm >= J.x_mt) {
                    J.cm = 0;
                    ++J.cj;
                    ++J.cg;
                }
            }
        };
        bool xbusy = false;
        for (uint32_t k = 0; k < nj; ++k) {
            xbusy = xbusy || (jobs[k].units && jobs[k].xon);
        }
#endif
        uint32_t it = 0;
#ifdef SE_DRISC_PROF  // (debug: cycles per section over the call, refreshes; at 448.. of the working region)
        uint64_t pf[6] = {0, 0, 0, 0, 0, 0};
        const uint64_t pf_t0 = get_timestamp();
#define PF_BEGIN const uint64_t pf_s = get_timestamp();
#define PF_END(i) pf[i] += get_timestamp() - pf_s;
#else
#define PF_BEGIN
#define PF_END(i)
#endif
#ifdef SE_DRISC_X
        while ((left || xbusy) && !stop) {
            xbusy = false;
            *word(496) += 1;  // (debug: passes)
#else
        while (left && !stop) {
#endif
            for (uint32_t k = 0; k < nj; ++k) {
                Job& J = jobs[k];
#ifndef SE_DRISC_STREAM0
#define SE_DRISC_STREAM0 0
#endif
                const uint8_t stream = (SE_DRISC_STREAM0 + k) & 1, noc = J.noc;
#if defined(SE_DRISC_XEMU) && SE_DRISC_XEMU_FIRST
#ifdef SE_DRISC_XEMU
                if (J.xslot && J.xpend) {
#if SE_DRISC_XEMU_OTHER
                    const uint8_t xs = stream ^ 1;
                    while (J.xpend && experimental::dma_get_reads_outstanding(xs) < SE_DRISC_XEMU_CAP) {
#else
                    const uint8_t xs = stream;
                    if (J.xt - J.xh >= SE_DRISC_XEMU_CAP) {
                        xrefresh(J, stream);
                    }
                    while (J.xpend && J.xt - J.xh < SE_DRISC_XEMU_CAP) {
#endif
                        const uint32_t upc2 = J.rg / J.ub, u2 = (J.rd + J.units / 2 + J.xi) % J.units;
                        const uint32_t c2 = u2 / upc2, b2 = (u2 % upc2) * J.ub;
                        const uint32_t src2 =
                            J.gddr + (J.loads[c2 / J.per_e] * J.per_e + c2 % J.per_e) * J.chunk + b2 * J.blk;
                        experimental::dma_async_read<false>(xs, src2, J.xslot, SE_DRISC_XEMU_B);
#if !SE_DRISC_XEMU_OTHER
                        J.xq[J.xt++ & 7] = J.tot++;
#endif
                        --J.xpend;
                        ++J.xi;
                    }
                }
#endif
#endif
                // DMA ahead into free slots (unit rd: chunk rd / RG of load (rd / RG) / PER_E, block rd % RG)
                {
                    PF_BEGIN
#ifdef SE_DRISC_FREE_SENT  // a slot is free once its unit's writes left L1 (sent), not when they are acked
                    const uint32_t freed = J.sn;
#else
                    const uint32_t freed = J.fl;
#endif
                    while (J.rd < J.units && J.rd - freed < J.ns &&
                           (J.dmax >= 255 || experimental::dma_get_reads_outstanding(stream) < J.dmax)) {
                        const uint32_t upc = J.rg / J.ub, c = J.rd / upc, b = (J.rd % upc) * J.ub;
                        const uint32_t src =
                            J.gddr + (J.loads[c / J.per_e] * J.per_e + c % J.per_e) * J.chunk + b * J.blk;
#if !defined(SE_W_NOREAD) && !defined(SE_GU_NOREAD)  // (perf probe: gate/up weights not read, garbage)
                        // (no ready poll: at most NS <= 15 reads are in flight, far below the stream's queue)
#if SE_DRISC_LEAN
                        lean_dma_read(stream, src, J.slot0 + (J.rd % J.ns) * J.ub * J.blk, J.dattr);
#else
                        experimental::dma_async_read<false>(
                            stream, src, J.slot0 + (J.rd % J.ns) * J.ub * J.blk, J.ub * J.blk);
#endif
                        ++dma_total[stream];
                        ++J.tot;
#else
                        (void)src;
#endif
#ifdef SE_DRISC_XEMU
#if SE_DRISC_XEMU_MODE == 0
                        if (++J.since >= SE_DRISC_XEMU) {
                            J.since = 0;
                            ++J.xpend;
                        }
#else
                        if (J.rd % (J.per_e * upc) == 0) {  // (a load's first unit)
                            const uint32_t ld = J.rd / (J.per_e * upc), nld = J.units / (J.per_e * upc);
#if SE_DRISC_XEMU_MODE == 1
                            J.xpend += SE_DRISC_XEMU_K;
#else
                            J.xpend +=
                                ld == 0 ? (nld > 1 ? 2 : 1) * SE_DRISC_XEMU_K : (ld + 1 < nld ? SE_DRISC_XEMU_K : 0);
#endif
                        }
#endif
#endif
                        ++J.rd;
                    }
                    PF_END(0)
                }
#if defined(SE_DRISC_XEMU) && !SE_DRISC_XEMU_FIRST
#ifdef SE_DRISC_XEMU
                if (J.xslot && J.xpend) {
#if SE_DRISC_XEMU_OTHER
                    const uint8_t xs = stream ^ 1;
                    while (J.xpend && experimental::dma_get_reads_outstanding(xs) < SE_DRISC_XEMU_CAP) {
#else
                    const uint8_t xs = stream;
                    if (J.xt - J.xh >= SE_DRISC_XEMU_CAP) {
                        xrefresh(J, stream);
                    }
                    while (J.xpend && J.xt - J.xh < SE_DRISC_XEMU_CAP) {
#endif
                        const uint32_t upc2 = J.rg / J.ub, u2 = (J.rd + J.units / 2 + J.xi) % J.units;
                        const uint32_t c2 = u2 / upc2, b2 = (u2 % upc2) * J.ub;
                        const uint32_t src2 =
                            J.gddr + (J.loads[c2 / J.per_e] * J.per_e + c2 % J.per_e) * J.chunk + b2 * J.blk;
                        experimental::dma_async_read<false>(xs, src2, J.xslot, SE_DRISC_XEMU_B);
#if !SE_DRISC_XEMU_OTHER
                        J.xq[J.xt++ & 7] = J.tot++;
#endif
                        --J.xpend;
                        ++J.xi;
                    }
                }
#endif
#endif
#ifdef SE_DRISC_DRAIN  // (every unit ready this pass, not one: the loop's other work no longer delays them)
#define SE_DRISC_STEP while
#else
#define SE_DRISC_STEP if
#endif
#ifdef SE_DRISC_FREE_SENT  // (transaction ids cycle over the 7 of the job, apart from the slots: <= 7 unacked units)
                SE_DRISC_STEP(
                    J.wr < J.rd && J.wr - J.fl < 7 && J.wr < J.rd - experimental::dma_get_reads_outstanding(stream)) {
#else
#if SE_DRISC_LEAN
                if (J.wr >= J.dn && J.wr < J.rd) {  // (the stream completes in order: rd - outstanding are done)
#if SE_XEMU_SAME
                    xrefresh(J, stream);
#else
                    J.dn = J.rd - experimental::dma_get_reads_outstanding(stream);
#endif
                }
                SE_DRISC_STEP(J.wr < J.dn) {
#elif SE_XEMU_SAME
                xrefresh(J, stream);
                SE_DRISC_STEP(J.wr < J.rd && J.wr < J.dn) {
#else
                SE_DRISC_STEP(J.wr < J.rd && J.wr < J.rd - experimental::dma_get_reads_outstanding(stream)) {
#endif
#endif
                    const uint32_t upc = J.rg / J.ub, c = J.wr / upc, b = (J.wr % upc) * J.ub;
                    const uint32_t j0 = b * J.G, j1 = (b + J.ub) * J.G;  // the unit's receivers
                    bool granted = true;
                    for (uint32_t j = j0; j < j1; ++j) {
                        granted = granted && J.cr[j] >= c + 1;
                    }
                    if (!granted) {  // refresh the unit's receivers' counters (NoC reads)
                        PF_BEGIN
                        const uint32_t cw = se_drisc_credit(k);
                        for (uint32_t j = j0; j < j1; ++j) {
                            noc_async_read(J.rgrant[j], base + cw + 16 * j, 4, noc);
                        }
                        noc_async_read_barrier(noc);
                        invalidate_l1_cache();
                        granted = true;
                        for (uint32_t j = j0; j < j1; ++j) {
                            J.cr[j] = *word(cw + 16 * j);
                            granted = granted && J.cr[j] >= c + 1;
                        }
                        PF_END(1)
#ifdef SE_DRISC_PROF
                        ++pf[5];
#endif
                    }
                    if (granted) {
                        PF_BEGIN
                        const uint32_t src0 = J.slot0 + (J.wr % J.ns) * J.ub * J.blk;
                        const uint32_t dst = J.land + (J.regions[c / J.per_e] * J.per_e + c % J.per_e) * J.blk;
#ifdef SE_DRISC_FREE_SENT
                        const uint32_t trid = J.trid0 + J.wr % 7;
#else
                        const uint32_t trid = J.trid0 + J.wr % J.ns;
#endif
#ifndef SE_W_NOFWD  // (perf probe: the weight blocks are not forwarded, only signalled)
                        for (uint32_t j = j0; j < j1; ++j) {  // receiver j gets block j / G (of the chunk)
                            const uint32_t src = src0 + (j / J.G - b) * J.blk;
                            if (J.blk <= NOC_MAX_BURST_SIZE) {
                                fast_write(noc, src, J.coord[j], dst, trid);
                            } else {
#if SE_DRISC_LEAN >= 3
                                fast_write_long(noc, src, J.coord[j], dst, J.blk);
                                continue;
#endif
                                for (uint32_t o = 0; o < J.blk; o += NOC_MAX_BURST_SIZE) {
                                    const uint32_t n = J.blk - o < NOC_MAX_BURST_SIZE ? J.blk - o : NOC_MAX_BURST_SIZE;
                                    noc_async_write_one_packet_with_trid(
                                        src + o, J.rx[j] | (dst + o), n, trid, write_cmd_buf, noc, fwd_vc);
                                }
                            }
                        }
#endif
#if SE_DRISC_LEAN >= 3
                        J.pm[J.wr % 16] = noc_posted_writes_num_issued[noc];
#endif
                        ++J.wr;
                        PF_END(2)
                    }
#ifdef SE_DRISC_DRAIN
                    else {
                        break;
                    }
#endif
                }
                PF_BEGIN
#if SE_DRISC_LEAN >= 3
                if (J.fl < J.wr) {  // (one counter read: every unit whose writes all left, in order)
                    const uint32_t sent = NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);
                    while (J.fl < J.wr && static_cast<int32_t>(sent - J.pm[J.fl % 16]) >= 0) {
                        const uint32_t upc = J.rg / J.ub, b = (J.fl % upc) * J.ub;
                        for (uint32_t j = b * J.G; j < (b + J.ub) * J.G; ++j) {
#if SE_DRISC_LEAN >= 4
                            fast_inc_nopoll(noc, J.coord[j]);
#else
                            fast_inc(noc, J.coord[j]);
#endif
                        }
                        ++J.fl;
                        --left;
                    }
                }
#else
#ifdef SE_DRISC_FREE_SENT
                const uint32_t fl_trid = J.trid0 + J.fl % 7;
                if (J.sn < J.wr && ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, J.trid0 + J.sn % 7)) {
                    ++J.sn;
                }
#else
                const uint32_t fl_trid = J.trid0 + J.fl % J.ns;
#endif
#ifdef SE_DRISC_DRAIN
#ifdef SE_DRISC_FREE_SENT
                while (J.fl < J.wr && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, J.trid0 + J.fl % 7)) {
#else
                while (J.fl < J.wr &&
                       ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, J.trid0 + J.fl % J.ns)) {
#endif
                    (void)fl_trid;
#else
                if (J.fl < J.wr && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, fl_trid)) {
#endif
                    const uint32_t upc = J.rg / J.ub, b = (J.fl % upc) * J.ub;
                    for (uint32_t j = b * J.G; j < (b + J.ub) * J.G; ++j) {
                        if (J.blk <= NOC_MAX_BURST_SIZE) {
#if SE_DRISC_LEAN >= 2
                            fast_inc_nopoll(noc, J.coord[j]);
#else
                            fast_inc(noc, J.coord[j]);
#endif
                        } else {
                            noc_semaphore_inc(J.rdata[j], 1, noc);
                        }
                    }
                    ++J.fl;
                    --left;
#ifdef SE_DRISC_FREE_SENT
                    J.sn = J.sn > J.fl ? J.sn : J.fl;  // (acked implies sent)
#endif
                }
#endif
                PF_END(3)
#ifdef SE_DRISC_X
                if (J.units && J.xon) {
                    // x: read the owned segments ahead (at most SE_DRISC_XS unwritten-or-uncounted), write the landed
                    // ones once their relays hold the chunk (credit: the relay's batches reserved), then - one
                    // flush check over every x write - count them into the relays' arrival words. Never blocks.
                    while (!J.cdone && J.xt - J.xf < SE_DRISC_XS) {
                        uint32_t t, j, o, q, r;
                        if (!x_next(J, t, j, o, q, r)) {
                            J.cdone = true;
                            break;
                        }
                        const uint32_t i = J.xt;
                        experimental::dma_async_read<false>(
                            stream,
                            J.x_xaddr + (t / J.x_banks) * J.x_page + j * J.x_seg,
                            J.xs0 + (i % SE_DRISC_XS) * J.x_seg,
                            J.x_seg);
                        J.xq[J.xt++ & 7] = J.tot++;
                        J.tq[i & 7] = q;
                        J.tr[i & 7] = r;
                        J.to_[i & 7] = o;
                        *word(508) += 1;  // (debug: x tasks)
                    }
                    if (J.xw < J.xt && J.xw >= J.xdone) {
                        xrefresh(J, stream);
                    }
                    bool refreshed = false;
                    while (J.xw < J.xdone) {
                        const uint32_t i = J.xw, o = J.to_[i & 7], q = J.tq[i & 7], r = J.tr[i & 7];
                        const uint32_t need = q / J.x_batch + 1;
                        bool ok = true;
                        for (uint32_t kk = 0; kk < J.x_nrect; ++kk) {
                            ok = ok && J.x_rsv[kk * J.x_vs + o] >= need;
                        }
                        if (!ok) {
                            *word(500) += 1;  // (debug: credit stalls)
                            if (refreshed || it % SE_DRISC_X_RPER != 0) {
                                break;  // (one credit read a pass, every RPER-th pass: the weights keep moving)
                            }
                            refreshed = true;
                            *word(504) += 1;  // (debug: credit reads)
                            for (uint32_t kk = 0; kk < J.x_nrect; ++kk) {
                                const uint32_t rl = kk * J.x_vs + o;
                                noc_async_read(J.x_rbase[rl] | J.x_rsva, J.xia + XI_RSV_OFF + 16 * rl, 4, J.noc);
                            }
                            noc_async_read_barrier(J.noc);
                            invalidate_l1_cache();
                            for (uint32_t kk = 0; kk < J.x_nrect; ++kk) {
                                const uint32_t rl = kk * J.x_vs + o;
                                J.x_rsv[rl] = *reinterpret_cast<volatile uint32_t*>(J.xia + XI_RSV_OFF + 16 * rl);
                            }
                            continue;
                        }
                        const uint32_t dst = J.x_ring + (q % J.x_rmc) * 32 * J.x_seg + r * J.x_seg;
                        for (uint32_t kk = 0; kk < J.x_nrect; ++kk) {
                            x_write(
                                noc, J.xs0 + (i % SE_DRISC_XS) * J.x_seg, J.x_rcoord[kk * J.x_vs + o], dst, J.x_seg);
                        }
                        if constexpr (write_reg_cmd_buf == write_cmd_buf) {  // (shared: the weights' registers back)
                            fast_write_state(noc, J.blk);
                        }
                        ++J.xw;
                    }
                    // (the command buffer drained first: an id reads done while its last write still sits in the
                    // buffer - TS_HANG_WORKLOG wave 14)
                    if (J.xf < J.xw && noc_cmd_buf_ready(noc, write_reg_cmd_buf) &&
                        ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, 15)) {
                        const uint32_t upto = J.xw;
                        for (uint32_t i = J.xf; i < upto; ++i) {
                            const uint32_t o = J.to_[i & 7], q = J.tq[i & 7];
                            for (uint32_t kk = 0; kk < J.x_nrect; ++kk) {
                                fast_inc_state(noc, J.x_arra[q % J.x_rmc]);
                                fast_inc(noc, J.x_rcoord[kk * J.x_vs + o]);
                            }
                        }
                        fast_inc_state(noc, J.dsem);  // (the weights' data semaphore increments' registers back)
                        J.xf = upto;
                    }
                    xbusy = xbusy || !J.cdone || J.xf < J.xt;
                }
#endif
            }
            if ((++it & 0xFFF) == 0) {
                for (uint32_t k = 0; k < nj; ++k) {
                    volatile uint32_t* st = word(se_drisc_state(k));
                    st[2] = jobs[k].rd;
                    st[3] = jobs[k].wr;
                    st[4] = jobs[k].fl;
                }
                invalidate_l1_cache();
                stop = *word(SE_DRISC_STOP) != 0;
            }
        }
#ifdef SE_DRISC_XEMU
        *word(504) += jobs[0].xi + (nj > 1 ? jobs[1].xi : 0);  // (debug: x reads issued, over the calls)
        for (uint32_t k = 0; k < nj; ++k) {  // (the extra reads drained: the next call's in-order count starts clean)
            if (jobs[k].xslot) {
                for (uint8_t s = 0; s < 2; ++s) {
                    while (experimental::dma_get_reads_outstanding(s)) {
                    }
                }
#if SE_DRISC_XEMU_W
                while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(jobs[k].noc, 15)) {
                }
#endif
            }
        }
#endif
        for (uint32_t k = 0; k < nj; ++k) {
            volatile uint32_t* st = word(se_drisc_state(k));
            st[2] = jobs[k].rd;
            st[3] = jobs[k].wr;
            st[4] = jobs[k].fl;
        }
        for (uint32_t q = 0; q < 2; ++q) {  // (debug: the stream's hardware read count since start vs ours)
            volatile uint32_t* st = word(se_drisc_state(q));
            st[6] = READ_TX_STREAM_REG(q, TX_REG_STREAM_TOTAL_READS_REG_OFFSET) - hw0[q];
            st[7] = dma_total[q];
        }
#ifdef SE_DRISC_PROF
        pf[4] = get_timestamp() - pf_t0;
        if (jobs[0].units) {  // (the last call with work: dma, refresh, write, flush+inc, total cycles, refreshes)
            for (uint32_t q = 0; q < 6; ++q) {
                *word(448 + 4 * q) = static_cast<uint32_t>(pf[q]);
            }
            *word(472) = jobs[0].units + (nj > 1 ? jobs[1].units : 0);
        }
#endif
        *word(SE_DRISC_DONE) = seen;
    }
    for (uint32_t k = 0; k < nj; ++k) {
        const uint8_t noc = get_arg_val<uint32_t>(3 + k);
        noc_async_write_set_trid(0, noc);
        noc_async_write_barrier(noc);
        noc_async_atomic_barrier(noc);
        noc_async_full_barrier(noc);
    }
    if constexpr (SE_DRISC_BURST != 0) {
        experimental::dma_restore_default_transfer_attrs();
    }
    if (mp_applied) {
        mpfe(0);
    }
    *word(SE_DRISC_READY) = 0;
}
