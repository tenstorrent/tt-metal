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
// RT: 0 working base (se_drisc.hpp layout), 1 working region bytes, 2 NJ, 3.. NJ x the job's NoC
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/gddr_dma.h"
#ifdef SE_DRISC_MPFE
#include "experimental/gddr_mc.h"
#endif
#include "se_drisc.hpp"

// the weight writes' unicast VC (as se10_fwd.cpp: off the default write VC the h / y traffic shares); SE_FWD_VC
#ifdef SE_FWD_VC
constexpr uint8_t fwd_vc = SE_FWD_VC;
#else
constexpr uint8_t fwd_vc = 3;
#endif

struct Job {
    uint32_t gddr, chunk, blk, rg, R, G, per_e, land, units, slot0, ns, trid0;
    uint32_t rd, wr, fl;
    uint8_t noc;
    uint8_t loads[SD_MAX_LOADS], regions[SD_MAX_LOADS];
    uint32_t cr[SE_DRISC_MAX_R];
    uint64_t rx[SE_DRISC_MAX_R], rdata[SE_DRISC_MAX_R], rgrant[SE_DRISC_MAX_R];
};

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0), region = get_arg_val<uint32_t>(1), nj = get_arg_val<uint32_t>(2);
    auto word = [&](uint32_t off) { return reinterpret_cast<volatile uint32_t*>(base + off); };
    for (uint32_t o = 0; o < se_drisc_desc(0); o += 4) {
        if (o != SE_DRISC_STOP) {
            *word(o) = 0;
        }
    }
    for (uint32_t k = 0; k < nj; ++k) {
        if (get_arg_val<uint32_t>(3 + k) != noc_index) {
            noc_local_state_init(get_arg_val<uint32_t>(3 + k));  // (the kernel start set up its own NoC only)
        }
    }
    *word(SE_DRISC_READY) = SE_DRISC_READY_MAGIC;
#ifdef SE_DRISC_MPFE  // GDDR arbitration weight of this tile's port (persists until reset; restored on exit)
    const uint32_t mpfe_port = SE_DRISC_MPFE_PORT, mpfe_old = gddr_mc_read_mpfe_weight(mpfe_port);
    gddr_mc_write_mpfe_weight(mpfe_port, SE_DRISC_MPFE);
#ifdef SE_DRISC_MPFE_ORD
    const uint32_t ord_old = gddr_mc_read_mpfe_weight(1);
    gddr_mc_write_mpfe_weight(1, SE_DRISC_MPFE_ORD);
#endif
#endif
#ifndef SE_DRISC_BURST
#define SE_DRISC_BURST 255
#endif
    if constexpr (SE_DRISC_BURST != 0) {  // (0: leave the DMA transfer attributes alone)
        experimental::dma_set_burst_size(SE_DRISC_BURST);
    }
    const uint32_t room = (region - SE_DRISC_SLOTS) / nj;
    uint32_t seen = 0, dma_total[2] = {0, 0};  // (debug: this kernel's DMA reads per stream)
    const uint32_t hw0[2] = {READ_TX_STREAM_REG(0, TX_REG_STREAM_TOTAL_READS_REG_OFFSET),
                             READ_TX_STREAM_REG(1, TX_REG_STREAM_TOTAL_READS_REG_OFFSET)};
    Job jobs[2];
    bool stop = false;
    while (!stop) {
        invalidate_l1_cache();
        if (*word(SE_DRISC_STOP)) {
            break;
        }
        bool posted = true;
        for (uint32_t k = 0; k < nj; ++k) {
            posted = posted && *word(se_drisc_jobs(k)) != seen;
        }
        if (!posted) {
            continue;
        }
        ++seen;
        uint32_t left = 0;
        for (uint32_t k = 0; k < nj; ++k) {
            Job& J = jobs[k];
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
                J.cr[j] = 0;
            }
            J.slot0 = base + SE_DRISC_SLOTS + k * room;
            J.ns = room / J.blk;
            J.ns = J.ns > 7 ? 7 : J.ns;  // (transaction ids 1-7 / 8-14 per job)
            J.trid0 = 1 + 7 * k;
            J.units = n_load * J.per_e * J.rg;
            J.rd = J.wr = J.fl = 0;
            left += J.units;
            volatile uint32_t* st = word(se_drisc_state(k));
            st[0] = seen;
            st[1] = J.units;
            st[5] = n_load;
        }
        uint32_t it = 0;
        while (left && !stop) {
            for (uint32_t k = 0; k < nj; ++k) {
                Job& J = jobs[k];
#ifndef SE_DRISC_STREAM0
#define SE_DRISC_STREAM0 0
#endif
                const uint8_t stream = (SE_DRISC_STREAM0 + k) & 1, noc = J.noc;
                // DMA ahead into free slots (unit rd: chunk rd / RG of load (rd / RG) / PER_E, block rd % RG)
#ifndef SE_DRISC_DMA_MAX
#define SE_DRISC_DMA_MAX 255
#endif
                while (J.rd < J.units && J.rd - J.fl < J.ns &&
                       (SE_DRISC_DMA_MAX >= 255 ||
                        experimental::dma_get_reads_outstanding(stream) < SE_DRISC_DMA_MAX)) {
                    const uint32_t c = J.rd / J.rg, b = J.rd % J.rg;
                    const uint32_t src = J.gddr + (J.loads[c / J.per_e] * J.per_e + c % J.per_e) * J.chunk + b * J.blk;
#ifndef SE_W_NOREAD  // (perf probe: weights not read, garbage)
                    experimental::dma_async_read(stream, src, J.slot0 + (J.rd % J.ns) * J.blk, J.blk);
                    ++dma_total[stream];
#else
                    (void)src;
#endif
                    ++J.rd;
                }
                if (J.wr < J.rd && J.wr < J.rd - experimental::dma_get_reads_outstanding(stream)) {
                    const uint32_t c = J.wr / J.rg, b = J.wr % J.rg;
                    bool granted = true;
                    for (uint32_t j = b * J.G; j < (b + 1) * J.G; ++j) {
                        granted = granted && J.cr[j] >= c + 1;
                    }
                    if (!granted) {  // refresh the block's receivers' counters (NoC reads)
                        const uint32_t cw = se_drisc_credit(k);
                        for (uint32_t j = b * J.G; j < (b + 1) * J.G; ++j) {
                            noc_async_read(J.rgrant[j], base + cw + 16 * j, 4, noc);
                        }
                        noc_async_read_barrier(noc);
                        invalidate_l1_cache();
                        granted = true;
                        for (uint32_t j = b * J.G; j < (b + 1) * J.G; ++j) {
                            J.cr[j] = *word(cw + 16 * j);
                            granted = granted && J.cr[j] >= c + 1;
                        }
                    }
                    if (granted) {
                        const uint32_t src = J.slot0 + (J.wr % J.ns) * J.blk;
                        const uint32_t dst = J.land + (J.regions[c / J.per_e] * J.per_e + c % J.per_e) * J.blk;
                        const uint32_t trid = J.trid0 + J.wr % J.ns;
                        for (uint32_t j = b * J.G; j < (b + 1) * J.G; ++j) {
                            for (uint32_t o = 0; o < J.blk; o += NOC_MAX_BURST_SIZE) {
                                const uint32_t n = J.blk - o < NOC_MAX_BURST_SIZE ? J.blk - o : NOC_MAX_BURST_SIZE;
                                noc_async_write_one_packet_with_trid(
                                    src + o, J.rx[j] | (dst + o), n, trid, write_cmd_buf, noc, fwd_vc);
                            }
                        }
                        ++J.wr;
                    }
                }
                if (J.fl < J.wr &&
                    ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, J.trid0 + J.fl % J.ns)) {
                    const uint32_t b = J.fl % J.rg;
                    for (uint32_t j = b * J.G; j < (b + 1) * J.G; ++j) {
                        noc_semaphore_inc(J.rdata[j], 1, noc);
                    }
                    ++J.fl;
                    --left;
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
#ifdef SE_DRISC_MPFE
    gddr_mc_write_mpfe_weight(mpfe_port, mpfe_old);
#ifdef SE_DRISC_MPFE_ORD
    gddr_mc_write_mpfe_weight(1, ord_old);
#endif
#endif
    *word(SE_DRISC_READY) = 0;
}
