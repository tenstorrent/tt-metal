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
// its own stream, which completes in order) into one of the job's NS L1 slots, then written to the G receivers that
// share block b (receiver j gets block j / G) at landing slot (region of c's load) * PER_E + c % PER_E, once each of
// them has granted chunk c, on the job's NoC with transaction id TRID0 + u % NS; when the id has no outstanding writes
// the receivers' data semaphores are bumped (in order: each receiver pushes its landing slots by count) and the slot
// refills. A receiver counts its grants in a semaphore of its own (zero at each launch); the sender reads the counters
// when it runs out of grants. The jobs advance round-robin.
//
// RT: 0 working base (se_drisc.hpp layout), 1 working region bytes, 2 NJ, 3.. NJ x the job's NoC, NJ x the job's DMA
//     stream (distinct), then the MPFE ports of this tile and of the bank's NOC0 / NOC1 endpoints
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/gddr_dma.h"
#include "experimental/gddr_mc.h"
#include "se_drisc.hpp"

// the weight writes' unicast VC (off the default write VC the h / y traffic shares)
constexpr uint8_t fwd_vc = 3;

constexpr uint8_t inc_vc = NOC_UNICAST_WRITE_VC;
constexpr uint32_t wr_resp = NOC_CMD_RESP_MARKED;

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
}
inline void fast_write(uint8_t noc, uint32_t src, uint32_t coord, uint32_t dst, uint32_t trid) {
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
}
// the unit's next receiver: the same destination address and transaction id as the write before it on this buffer
inline void fast_write_next(uint8_t noc, uint32_t src, uint32_t coord) {
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
}
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

// Probes (flat_drisc.cpp, FlatDriscProbe): SE_DRISC_PROF counts the loop's cycles (rdcycle) and events into
// SE_DRISC_PROF_OFF: 0 calls, 1 loop cycles (sum), 3 a cycle stamp while idle, 4 the probe, 5 the last call's cycles,
// 8 + 24 k + SP_* job k's counters. SE_DRISC_PROBE: 2 DMA only (no grant checks, no weight writes), 3 writes only (no
// DMA), 4 neither (the loop and the semaphore increments)
#ifndef SE_DRISC_PROF
#define SE_DRISC_PROF 0
#endif
#ifndef SE_DRISC_PROBE
#define SE_DRISC_PROBE 0
#endif
constexpr bool kSteps = SE_DRISC_PROF >= 2;  // (per-step counters)
constexpr bool kDma = SE_DRISC_PROBE != 3 && SE_DRISC_PROBE != 4;
constexpr bool kWrite = SE_DRISC_PROBE != 2 && SE_DRISC_PROBE != 4;
enum : uint32_t {
    SP_C_DMA,         // cycles in the DMA issue step
    SP_C_WR,          // cycles in the write step (landed check, grants, writes)
    SP_C_FL,          // cycles in the flush step (ack poll, semaphore increments)
    SP_C_GRANT,       // cycles refreshing grants (NoC reads of the receivers' counters, blocking)
    SP_C_WISSUE,      // cycles issuing a unit's NoC writes
    SP_N_DMA,         // units DMA'd
    SP_N_WR,          // units written
    SP_N_FL,          // units flushed
    SP_N_GRANT,       // grant refreshes
    SP_N_GRANT_FAIL,  // refreshes that still found a receiver without the grant
    SP_WAIT_DMA,      // iterations with a unit DMA'd but not landed (and none to write)
    SP_WAIT_ACK,      // iterations with a written unit not acked
    SP_FULL,          // iterations with every L1 slot taken (no DMA possible)
    SP_CAPBLK,        // iterations with a free slot but the DMA cap reached
    SP_IDLE,          // iterations that did nothing
    SP_ITER,          // iterations
    SP_C_DMA_LAT,     // cycles from a unit's DMA issue to its write's start (landed, granted, the loop's turn)
    SP_C_ACK_LAT,     // cycles from a unit's writes to its flush (acked, the loop's turn)
    SP_N
};
static_assert(SP_N <= 24, "probe counters: 24 words a job (SE_DRISC_PROF_OFF + 8 + 24 k)");
// the next unit of job J after cursor q
#define SE_CUR_NEXT(J, q)                \
    do {                                 \
        if (++(q).uc == (J).upc) {       \
            (q).uc = 0;                  \
            ++(q).c;                     \
            if (++(q).ce == (J).per_e) { \
                (q).ce = 0;              \
                ++(q).e;                 \
            }                            \
        }                                \
        if (++(q).s == (J).ns) {         \
            (q).s = 0;                   \
        }                                \
        if (++(q).t == 7) {              \
            (q).t = 0;                   \
        }                                \
    } while (0)

inline uint32_t cyc() {
    if constexpr (SE_DRISC_PROF) {
        uint32_t c;
        asm volatile("rdcycle %0" : "=r"(c));
        return c;
    } else {
        return 0;
    }
}

// A unit's place in a job's stream (no divisions in the hot loop): the unit in its chunk, the chunk in its load, the
// load, the chunk's index in the call, the unit's L1 slot / transaction id
struct Cur {
    uint32_t uc, ce, e, c, s, t;  // (t: the unit's transaction id index, unit mod 7)
};

struct Job {
    uint32_t gddr, chunk, blk, rg, R, G, per_e, land, units, slot0, ns, trid0, ub, dmax, attrs;
    uint32_t rd, wr, fl;  // units DMA'd, written, flushed (acked)
    uint32_t upc, ubb;    // units a chunk, bytes a unit
    Cur crd, cwr, cfl;    // the next unit to DMA / write / flush
    uint32_t sn, tsn;     // early free: units whose writes have left L1, the next one's transaction id index
    bool early;
    uint8_t noc, stream;
    uint8_t loads[SD_MAX_LOADS], regions[SD_MAX_LOADS];
    uint32_t cr[SE_DRISC_MAX_R];
    uint64_t rx[SE_DRISC_MAX_R], rdata[SE_DRISC_MAX_R], rgrant[SE_DRISC_MAX_R];
    uint32_t coord[SE_DRISC_MAX_R];  // (NOC_*_ADDR_COORDINATE of receiver j)
    uint32_t dsem;
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
    *word(SE_DRISC_READY) = SE_DRISC_READY_MAGIC;
    // GDDR MPFE weights (the job descriptor's word, per call): this tile's port, the bank's NOC0 / NOC1 endpoints'
    // ports (RT 3 + 2 NJ ..); the weights as found are restored when the call asks for none and on exit (they persist
    // until a reset)
    const uint32_t mp_port[3] = {
        get_arg_val<uint32_t>(3 + 2 * nj), get_arg_val<uint32_t>(4 + 2 * nj), get_arg_val<uint32_t>(5 + 2 * nj)};
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
    experimental::dma_set_burst_size(255);
    uint32_t seen_real[2] = {0, 0}, seen_skip[2] = {0, 0};
    uint32_t seen = 0, dma_total[2] = {0, 0};  // (debug: this kernel's DMA reads per stream)
    const uint32_t hw0[2] = {
        READ_TX_STREAM_REG(0, TX_REG_STREAM_TOTAL_READS_REG_OFFSET),
        READ_TX_STREAM_REG(1, TX_REG_STREAM_TOTAL_READS_REG_OFFSET)};
    Job jobs[2];
    [[maybe_unused]] uint32_t prof[2][SP_N] = {};
    [[maybe_unused]] uint32_t tsd[2][8], tsw[2][8];  // (probes: a slot's DMA issue / write stamps)
    if constexpr (SE_DRISC_PROF) {
        volatile uint32_t* pw = word(SE_DRISC_PROF_OFF);
        for (uint32_t i = 0; i < 64; ++i) {
            pw[i] = 0;
        }
        pw[4] = SE_DRISC_PROBE;
        if constexpr (kSteps) {  // (cycles a 64 back-to-back ops, x 16: rdcycle, NIU status read, DMA status read,
                                 // command buffer register write, L1 read, udiv)
            volatile uint32_t sink = 0;
            volatile uint32_t dv = 7, dn = 123456789;
            uint32_t t = cyc();
            for (uint32_t i = 0; i < 64; ++i) {
                sink += cyc();
            }
            pw[56] = (cyc() - t) * 16 / 64;
            t = cyc();
            for (uint32_t i = 0; i < 64; ++i) {
                sink += NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(15));
            }
            pw[57] = (cyc() - t) * 16 / 64;
            t = cyc();
            for (uint32_t i = 0; i < 64; ++i) {
                sink += experimental::dma_get_reads_outstanding(0);
            }
            pw[58] = (cyc() - t) * 16 / 64;
            t = cyc();
            for (uint32_t i = 0; i < 64; ++i) {
                NOC_CMD_BUF_WRITE_REG(noc_index, write_reg_cmd_buf, NOC_TARG_ADDR_LO, i);
            }
            pw[59] = (cyc() - t) * 16 / 64;
            t = cyc();
            for (uint32_t i = 0; i < 64; ++i) {
                sink += *word(SE_DRISC_STOP);
            }
            pw[60] = (cyc() - t) * 16 / 64;
            t = cyc();
            for (uint32_t i = 0; i < 64; ++i) {
                sink += dn / dv;
            }
            pw[61] = (cyc() - t) * 16 / 64;
            t = cyc();
            for (uint32_t i = 0; i < 64; ++i) {
                while (!noc_cmd_buf_ready(noc_index, write_reg_cmd_buf));
            }
            pw[62] = (cyc() - t) * 16 / 64;
        }
    }
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
            if constexpr (SE_DRISC_PROF) {
                *word(SE_DRISC_PROF_OFF + 12) = cyc();  // (the host's clock calibration: two reads, seconds apart)
            }
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
        const uint32_t room = n_real ? (region - SE_DRISC_SLOTS) / n_real : 0;  // (L1 slots split over the real jobs)
        uint32_t left = 0, ri = 0;
        for (uint32_t k = 0; k < nj; ++k) {
            Job& J = jobs[k];
            J.noc = get_arg_val<uint32_t>(3 + k);
            J.stream = get_arg_val<uint32_t>(3 + nj + k);
            if (!real[k]) {
                J.units = 0;
                J.rd = J.wr = J.fl = 0;
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
            J.dmax = d[SD_DMA_CAP] ? d[SD_DMA_CAP] : 255;  // (se_drisc.hpp SD_DMA_CAP)
            for (uint32_t l = 0; l < n_load; ++l) {
                const uint32_t v = d[SD_LOADS + l];
                J.loads[l] = v & 0xFF;
                J.regions[l] = (v >> 8) & 0xFF;
            }
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
            // 16 slots on one tile per bank M 64 57.8 -> 41.2 us) while they leave two units in flight a job, else
            // single blocks (K3's 13.5 KB blocks: one pair a job in half the L1; Galaxy load, one tile a bank
            // 1720 -> 1600 us; GLM-Flash's 9 KB blocks keep pairs: singles 413 -> 545 us). SD_UB: a caller's choice
            // (one whose unit does not fit the job's L1 falls back to single blocks)
            {
                const uint32_t ub = d[SD_UB] & 0xFF;
                if (ub) {
                    J.ub = J.rg % ub == 0 && room >= ub * J.blk ? ub : 1;
                } else {
                    J.ub = n_real > 1 && J.rg % 2 == 0 && room >= 4 * J.blk ? 2 : 1;
                }
            }
            J.ns = room / (J.ub * J.blk);
            constexpr uint32_t ns_max = 7;  // (transaction ids 1-7 / 8-14 per job)
            J.ns = J.ns > ns_max ? ns_max : J.ns;
            J.upc = J.rg / J.ub;
            J.ubb = J.ub * J.blk;
            J.crd = J.cwr = J.cfl = Cur{0, 0, 0, 0, 0, 0};
            J.sn = J.tsn = 0;
            // (early free for block pairs and up: GLM-Flash one tile a bank 378 -> 371 us; single blocks (K3) lose
            // more to the extra poll a pass than the earlier refill gains, 1380 -> 1404)
            J.early = (d[SD_UB] & SD_UB_EARLY_FREE) != 0 && J.ub >= 2;
            {  // (the DMA's transfer attributes: dma_async_read's, for a unit)
                DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
                attrs.f.transfer_size_words = (J.ub * J.blk) >> 4;
                attrs.f.transfer_start_read = 1;
                J.attrs = attrs.val;
            }
            J.trid0 = 1 + 7 * k;
            J.units = n_load * J.per_e * (J.rg / J.ub);
            J.rd = J.wr = J.fl = 0;
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
        for (uint32_t k = 0; k < nj; ++k) {  // (the fast path's constant registers, per NoC; the DMA source high word)
            if (jobs[k].units) {
                WRITE_TX_STREAM_REG(jobs[k].stream, TX_REG_STREAM_READ_TRANSFER_SOURCE_HIGH_REG_OFFSET, 0);
            }
            if (jobs[k].units && jobs[k].blk <= NOC_MAX_BURST_SIZE) {
                fast_write_state(jobs[k].noc, jobs[k].blk);
                fast_inc_state(jobs[k].noc, jobs[k].dsem);
            }
        }
        uint32_t it = 0;
        const uint32_t t_call0 = cyc();
        while (left && !stop) {
            for (uint32_t k = 0; k < nj; ++k) {
                Job& J = jobs[k];
                [[maybe_unused]] uint32_t* P = prof[k];
                const uint8_t stream = J.stream, noc = J.noc;
                [[maybe_unused]] bool did = false;
                // (flushes first: a slot the ack frees is refilled in the same pass)
                [[maybe_unused]] const uint32_t t2 = kSteps ? cyc() : 0;
                if (J.fl < J.wr &&
                    (!kWrite || ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, J.trid0 + J.cfl.t))) {
                    const uint32_t b = J.cfl.uc * J.ub;
                    if constexpr (kSteps) {
                        P[SP_C_ACK_LAT] += cyc() - tsw[k][J.cfl.s];
                    }
                    SE_CUR_NEXT(J, J.cfl);
                    for (uint32_t j = b * J.G; j < (b + J.ub) * J.G; ++j) {
                        if (J.blk <= NOC_MAX_BURST_SIZE) {
                            fast_inc(noc, J.coord[j]);
                        } else {
                            noc_semaphore_inc(J.rdata[j], 1, noc);
                        }
                    }
                    ++J.fl;
                    --left;
                    if constexpr (kSteps) {
                        ++P[SP_N_FL];
                        did = true;
                    }
                } else if constexpr (kSteps) {
                    P[SP_WAIT_ACK] += J.fl < J.wr;
                }
                // early free: a unit's slot refills once its writes have left L1 (the ack still gates its increments)
                if (J.early && J.sn < J.wr &&
                    (!kWrite || ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, J.trid0 + J.tsn))) {
                    ++J.sn;
                    if (++J.tsn == 7) {
                        J.tsn = 0;
                    }
                }
                [[maybe_unused]] const uint32_t t0 = kSteps ? cyc() : 0;
                // units landed: the stream's reads complete in order (one status read a pass: for the DMA cap and
                // the writes)
                // (none in flight: none outstanding; a skipped job has none - and no stream)
                const uint32_t landed =
                    kDma && J.rd != J.wr ? J.rd - experimental::dma_get_reads_outstanding(stream) : J.rd;
                // DMA ahead into free slots (unit rd: chunk rd / RG of load (rd / RG) / PER_E, block rd % RG)
                {
                    const uint32_t freed = J.early ? J.sn : J.fl;
                    if constexpr (kSteps) {
                        P[SP_FULL] += J.rd < J.units && J.rd - freed >= J.ns;
                    }
                    while (J.rd < J.units && J.rd - freed < J.ns && J.rd - landed < J.dmax) {
                        if constexpr (kDma) {
                            const Cur& q = J.crd;
                            const uint32_t src = J.gddr + (J.loads[q.e] * J.per_e + q.ce) * J.chunk + q.uc * J.ubb;
                            // (no ready poll: at most NS <= 15 reads are in flight, far below the stream's queue;
                            // the source's high word is zero, set at the call's start)
                            lean_dma_read(stream, src, J.slot0 + q.s * J.ubb, J.attrs);
                        }
                        SE_CUR_NEXT(J, J.crd);
                        ++dma_total[stream];
                        if constexpr (kSteps) {
                            tsd[k][J.crd.s == 0 ? J.ns - 1 : J.crd.s - 1] = cyc();
                        }
                        ++J.rd;
                        if constexpr (kSteps) {
                            ++P[SP_N_DMA];
                            did = true;
                        }
                    }
                    if constexpr (kSteps) {
                        P[SP_CAPBLK] += J.rd < J.units && J.rd - freed < J.ns;
                    }
                }
                [[maybe_unused]] const uint32_t t1 = kSteps ? cyc() : 0;
                if (J.wr < (kDma ? landed : J.rd)) {
                    const Cur& q = J.cwr;
                    const uint32_t c = q.c, b = q.uc * J.ub;
                    const uint32_t j0 = b * J.G, j1 = (b + J.ub) * J.G;  // the unit's receivers
                    bool granted = true;
                    if constexpr (kWrite) {
                        for (uint32_t j = j0; j < j1; ++j) {
                            granted = granted && J.cr[j] >= c + 1;
                        }
                    }
                    if (!granted) {  // refresh the unit's receivers' counters (NoC reads)
                        [[maybe_unused]] const uint32_t tg = kSteps ? cyc() : 0;
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
                        if constexpr (kSteps) {
                            P[SP_C_GRANT] += cyc() - tg;
                            ++P[SP_N_GRANT];
                            P[SP_N_GRANT_FAIL] += !granted;
                        }
                    }
                    if (granted) {
                        [[maybe_unused]] const uint32_t tw = kSteps ? cyc() : 0;
                        if constexpr (kWrite) {
                            const uint32_t src0 = J.slot0 + q.s * J.ubb;
                            const uint32_t dst = J.land + (J.regions[q.e] * J.per_e + q.ce) * J.blk;
                            const uint32_t trid = J.trid0 + q.t;
                            uint32_t src = src0, g = 0;
                            for (uint32_t j = j0; j < j1; ++j) {  // receiver j gets block j / G (of the chunk)
                                if (g == J.G) {
                                    g = 0;
                                    src += J.blk;
                                }
                                ++g;
                                if (J.blk <= NOC_MAX_BURST_SIZE) {
                                    if (j == j0) {
                                        fast_write(noc, src, J.coord[j], dst, trid);
                                    } else {
                                        fast_write_next(noc, src, J.coord[j]);
                                    }
                                } else {
                                    for (uint32_t o = 0; o < J.blk; o += NOC_MAX_BURST_SIZE) {
                                        const uint32_t n =
                                            J.blk - o < NOC_MAX_BURST_SIZE ? J.blk - o : NOC_MAX_BURST_SIZE;
                                        noc_async_write_one_packet_with_trid(
                                            src + o, J.rx[j] | (dst + o), n, trid, write_cmd_buf, noc, fwd_vc);
                                    }
                                }
                            }
                        }
                        if constexpr (kSteps) {
                            const uint32_t now = cyc();
                            P[SP_C_DMA_LAT] += now - tsd[k][J.cwr.s];
                            tsw[k][J.cwr.s] = now;
                        }
                        ++J.wr;
                        SE_CUR_NEXT(J, J.cwr);
                        if constexpr (kSteps) {
                            P[SP_C_WISSUE] += cyc() - tw;
                            ++P[SP_N_WR];
                            did = true;
                        }
                    }
                } else if constexpr (kSteps) {
                    P[SP_WAIT_DMA] += J.wr < J.rd;
                }
                if constexpr (kSteps) {
                    const uint32_t t3 = cyc();
                    P[SP_C_FL] += t0 - t2;
                    P[SP_C_DMA] += t1 - t0;
                    P[SP_C_WR] += t3 - t1;
                    P[SP_IDLE] += !did;
                    P[SP_ITER] += 1;
                }
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
        *word(SE_DRISC_DONE) = seen;
        if constexpr (SE_DRISC_PROF) {
            const uint32_t t = cyc() - t_call0;
            volatile uint32_t* pw = word(SE_DRISC_PROF_OFF);
            pw[0] += 1;
            pw[1] += t;
            pw[5] = t;
            for (uint32_t k = 0; k < 2; ++k) {
                for (uint32_t i = 0; i < SP_N; ++i) {
                    pw[8 + 24 * k + i] = prof[k][i];
                }
            }
        }
    }
    for (uint32_t k = 0; k < nj; ++k) {
        const uint8_t noc = get_arg_val<uint32_t>(3 + k);
        noc_async_write_set_trid(0, noc);
        noc_async_write_barrier(noc);
        noc_async_atomic_barrier(noc);
        noc_async_full_barrier(noc);
    }
    experimental::dma_restore_default_transfer_attrs();
    if (mp_applied) {
        mpfe(0);
    }
    *word(SE_DRISC_READY) = 0;
}
