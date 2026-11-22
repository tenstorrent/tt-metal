// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC gate/up weight senders (MIMO_FL_DRISC): the job of the Tensix weight readers + forwarders (se_reader.cpp +
// se10_fwd.cpp) on the DRAM cores (Blackhole programmable DRAM tiles, 2 per bank). A persistent kernel per DRISC
// (se_drisc_gu.cpp) is launched once per device (flat_drisc.cpp) and serves every flat call: each call, reader slot
// r's Tensix core (se_drisc_go.cpp) builds the call's schedule and writes a job descriptor into its DRISC's L1, then
// bumps the DRISC's job counter. The gate/up receivers grant landing slots straight to the DRISC (credit words in its
// L1); the DRISC bumps their data semaphores as blocks land, as the forwarder did.
//
// DRISC L1 layout (offsets from the kernel's working base, host: flat_drisc.cpp):
#pragma once
#include <stdint.h>

constexpr uint32_t SE_DRISC_STOP = 16;   // host: nonzero = exit
constexpr uint32_t SE_DRISC_READY = 32;  // kernel: its magic once it has zeroed its words (host waits on it)
constexpr uint32_t SE_DRISC_DONE = 48;   // kernel: calls completed (debug)
constexpr uint32_t SE_DRISC_MAX_R = 8;
constexpr uint32_t SE_DRISC_SLOTS = 2048;  // block slots from here to the end of the working region (split by job)
constexpr uint32_t SE_DRISC_READY_MAGIC = 0xD815C0DE;
// per job k (a tile serves 1 or 2 reader slots): jobs posted (Tensix atomic increments), its receivers' grants as last
// read (NoC read landing, + 16 j), progress (debug: call, units, read, written, flushed, loads), the job descriptor
constexpr uint32_t se_drisc_jobs(uint32_t k) { return k ? 64 : 0; }
// skipped jobs (a call whose slot another DRISC serves: an atomic increment, no descriptor)
constexpr uint32_t se_drisc_skips(uint32_t k) { return se_drisc_jobs(k) + 4; }
constexpr uint32_t se_drisc_credit(uint32_t k) { return 128 + 128 * k; }
constexpr uint32_t se_drisc_state(uint32_t k) { return 384 + 32 * k; }
constexpr uint32_t se_drisc_desc(uint32_t k) { return 512 + 640 * k; }

// descriptor words
constexpr uint32_t SD_GDDR = 0;      // this slot's weight region in the bank (bank-local address)
constexpr uint32_t SD_CHUNK = 1;     // chunk bytes (RG blocks)
constexpr uint32_t SD_BLK = 2;       // block bytes (one receiver's share of a chunk)
constexpr uint32_t SD_RG = 3;        // blocks per chunk
constexpr uint32_t SD_R = 4;         // receivers
constexpr uint32_t SD_G = 5;         // receivers per block (M-groups)
constexpr uint32_t SD_PER_E = 6;     // chunks per expert load
constexpr uint32_t SD_LAND = 7;      // landing ring address on the receivers
constexpr uint32_t SD_DATA_SEM = 8;  // the receivers' data semaphore address
constexpr uint32_t SD_N_LOAD = 9;    // gate/up loads this call
constexpr uint32_t SD_GRANTS = 10;   // the receivers' grant counter address (a semaphore, counts this call's grants)
constexpr uint32_t SD_RXY = 11;      // R receiver xy (x << 16 | y)
constexpr uint32_t SD_MPFE =
    SD_RXY + SE_DRISC_MAX_R;  // GDDR MPFE weights (job 0): 0 untouched, else bit 31 + the
                              // DRISC port's (bits 0-2), the NOC0 / NOC1 endpoints' (4-6 / 8-10)
// 1: both of the bank's DRISC tiles send this call: at most 2 DMA reads in flight each (two tiles with deep DMA queues
// next to the bank's NoC DRAM reads wedged the NoC: pure mode on K2.7, 3 of 3 runs; a 2-read cap or one tile: clean)
constexpr uint32_t SD_PAIRED = SD_MPFE + 1;
constexpr uint32_t SD_XON = SD_MPFE + 2;    // 1: this call's x comes from the DRISCs (SE_DRISC_X; the x info below)
constexpr uint32_t SD_LOADS = SD_MPFE + 3;  // N_LOAD x (expert | region << 8)
constexpr uint32_t SD_MAX_LOADS = 128;
static_assert((SD_LOADS + SD_MAX_LOADS) * 4 <= 640 && se_drisc_desc(2) <= SE_DRISC_SLOTS);

// x from the DRISCs (SE_DRISC_X): each DRISC reads the x row segments in its own bank and writes them into the relays'
// row-major chunks; per job an x area of SE_DRISC_XA bytes at the top of the working region (job k: region - (k + 1)
// XA): the launcher's x info (words XI_*), the sub-block's token ids (XI_IDX_OFF), the relays' credits read back
// (XI_RSV_OFF, 16 B each). On the relays, program semaphores (zero at every launch, before any kernel runs: no state
// across calls): SE_XSEM_RSV its credit (row-major batches reserved this call), SE_XSEM_ARR[p] the rows landed in chunk
// slot p.
constexpr uint32_t SE_DRISC_XA = 4096;
constexpr uint32_t XI_XADDR = 1, XI_PAGE = 2, XI_MT = 3, XI_NSB = 4, XI_VSTRIDE = 5, XI_NRECT = 6, XI_RMC = 7,
                   XI_BATCH = 8, XI_SEG = 9, XI_RING = 10, XI_RSV = 11, XI_SLOT = 12, XI_BANKS = 13, XI_SPB = 14,
                   XI_TIDX0 = 15, XI_TIDX1 = 17, XI_RXY = 19, XI_MAX_RL = 8, XI_ARR = 27, XI_MAX_RMC = 5, XI_NACT = 32,
                   XI_ENT = 33;
constexpr uint32_t XI_IDX_OFF = 1024, XI_RSV_OFF = 1536, XI_INFO_BYTES = 1024;
// the relays' semaphore ids (ids the relay kernels leave alone: 4 / 7 / 8 / 10 helper data, 5 helper credit, 9 XARR,
// 13 WORD, 3 / 14 / 15 the X2 / XFWD probes)
constexpr uint32_t SE_XSEM_RSV = 0;
constexpr uint32_t SE_XSEM_ARR[XI_MAX_RMC] = {1, 2, 6, 11, 12};
constexpr uint32_t SE_DRISC_XS = 8;  // x segment slots a job (<= the 8-entry DMA ring)
static_assert((XI_ENT + 2 * 95) * 4 <= XI_INFO_BYTES, "x info: up to 95 schedule entries");

// The receivers count their grants in their own L1 (no NoC writes from them: they are the x multicast's
// destinations); the DRISC reads the counters when it runs out of grants.
//
// The DRISC L1 as the NoC sees it: above the 8 GB GDDR window (bit 37: the high address register's bit 5), which
// reaches L1 whatever mode the tile's NIU is in. The stock NoC calls mask the high register to the PCIe bits, so the
// Tensix side issues these by hand.
constexpr uint32_t SE_DRISC_L1_MID = 0x20;

#if !defined(COMPILE_FOR_DRISC) && !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

inline uint32_t se_drisc_coord(uint32_t xy) {
    return (uint32_t)(NOC_XY_ADDR(xy >> 16, xy & 0xFFFF, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

// One write of at most NOC_MAX_BURST_SIZE bytes from local L1 into DRISC L1 (counted as an ordinary non-posted write:
// noc_async_write_barrier covers it). NOC0 only: tile 1 of each bank is its NOC1 endpoint, whose NOC1 NIU is in
// NOC2AXI mode and does not reach L1 (NOC0's NIU is in stream mode on every DRISC tile).
template <uint32_t noc = 0>
inline void se_drisc_write(uint32_t src, uint32_t xy, uint32_t addr, uint32_t bytes) {
    static_assert(noc == 0 && noc_index == 0, "DRISC L1 over NOC0 (this RISC's NOC: its write counters)");
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, SE_DRISC_L1_MID);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, se_drisc_coord(xy));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, bytes);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
    // (the stock NoC calls never write the high register - they assume it zero: a 0x20 left here sends this RISC's
    // later writes on this buffer, in this kernel or the core's next program, to the DRISC L1 window of their target -
    // e.g. the next program's DRAM writes into a DRISC's L1. Back to zero once the command has left the buffer)
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, 0);
}

// Atomic increment of a DRISC L1 word (non-posted: noc_async_atomic_barrier covers it). NOC0 only, as the write.
template <uint32_t noc = 0>
inline void se_drisc_inc(uint32_t xy, uint32_t addr, uint32_t incr) {
    static_assert(noc == 0 && noc_index == 0, "DRISC L1 over NOC0 (this RISC's NOC: its atomic counters)");
    while (!noc_cmd_buf_ready(noc, write_at_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_TARG_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_TARG_ADDR_MID, SE_DRISC_L1_MID);
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_TARG_ADDR_COORDINATE, se_drisc_coord(xy));
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_at_cmd_buf,
        NOC_CTRL,
        NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED | NOC_CMD_AT);
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_at_cmd_buf,
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(31) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_AT_DATA, incr);
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_CMD_CTRL, 0x1);
    noc_nonposted_atomics_acked[noc] += 1;
    while (!noc_cmd_buf_ready(noc, write_at_cmd_buf));  // (the high register back to zero: as se_drisc_write)
    NOC_CMD_BUF_WRITE_REG(noc, write_at_cmd_buf, NOC_TARG_ADDR_MID, 0);
}
#endif
