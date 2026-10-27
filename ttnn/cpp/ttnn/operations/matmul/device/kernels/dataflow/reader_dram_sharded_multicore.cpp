// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data movement for the multi-core DRAM-sharded decode matmul (num_workers_per_dram_bank >= 2; the DSMC_ defines
// are this variant's). The same source
// is built twice:
//   stream 0 (NOC_0): streams weight blocks into dfb::w.
//   stream 1 (NOC_1, DSMC_STREAM1): gathers the activation rows this core needs from their L1
//     shards, streams weight blocks too when DSMC_STREAMS_W is set (Blackhole: the two streams take
//     alternate K blocks), runs the in-group reduce-scatter (DSMC_REDUCE) and writes the finished
//     output tiles straight into the width-sharded output.
//
// Work item wi = ((bank * CM + cm) * CN + cn) * CK + kidx owns weight columns [n0, n1),
// K rows [k0, k1) and M rows [cm * MT, (cm + 1) * MT). Ranges follow the factory, so no per-core
// runtime arguments are needed; the only per-core table is DSMC_ROLE (one byte per logical core:
// 0 = idle, else wi + 1), which lets the program run on the bounding rectangle of its cores.
//
// Device 2.0 APIs throughout: Noc reads/writes with BankView (weight), UnicastEndpoint
// (activation and output shards, group partials) and CoreLocalMem / DataflowBuffer (local L1),
// NocOptions::TXN_ID reads and barriers, Noc::is_read_trid_flushed for the in-order retire, and
// Semaphore for the reduce. The read-command readiness check prevents completion counters
// from being sampled before submission; the sticky transaction tag is cleared on exit.

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "internal/tensor/transfer_noc_addr.h"
#include "experimental/kernel_args.h"
#ifdef DSMC_REDUCE
#include "api/semaphore.h"
#endif

namespace {

constexpr uint32_t MT = get_arg(args::MT);    // output tile rows
constexpr uint32_t Nbt = get_arg(args::Nbt);  // weight shard width per bank (tiles)
constexpr uint32_t KT = get_arg(args::KT);    // K (tiles)
constexpr uint32_t KB = get_arg(args::KB);    // K rows per streamed block
constexpr uint32_t CM = get_arg(args::CM);
constexpr uint32_t CN = get_arg(args::CN);      // column groups per bank
constexpr uint32_t CK = get_arg(args::CK);      // row groups per column group (CK > 1: reduce-scatter)
constexpr uint32_t P = get_arg(args::P);        // column passes per core
constexpr uint32_t NCP = get_arg(args::NCP);    // columns per pass slot (weight block width in L1)
constexpr uint32_t D = get_arg(args::D);        // weight blocks in flight per stream (trids 1..D)
constexpr uint32_t WT = get_arg(args::WT);      // weight tile bytes (DRAM-aligned)
constexpr uint32_t NGRP = get_arg(args::NGRP);  // column groups over all banks (rotation spread)
constexpr uint32_t ROT = get_arg(args::ROT);
constexpr uint32_t GX = get_arg(args::GX);  // worker grid width (DSMC_ROLE row stride)
constexpr uint32_t PAR = get_arg(args::PAR);
constexpr uint32_t WSINGLE = get_arg(args::WSINGLE);  // 1: stream 0 reads every block
constexpr uint32_t BT = KB * NCP;                     // dfb::w entries per block slot
static_assert(D >= 1 && D <= 7, "weight trids 1..D must stay below the activation trids 8..15");

constexpr uint8_t kRole[] = {DSMC_ROLE};

#ifdef DSMC_STREAM1
constexpr uint32_t XT = get_arg(args::XT);      // activation tile bytes
constexpr uint32_t SW = get_arg(args::SW);      // activation shard width (tiles)
constexpr uint32_t XL = get_arg(args::XL);      // activation blocks in flight ahead of the weight stream
constexpr uint32_t OT = get_arg(args::OT);      // output tile bytes
constexpr uint32_t OW = get_arg(args::OW);      // output shard width (tiles)
constexpr uint32_t OCAP = get_arg(args::OCAP);  // output shard capacity over all shards (tiles)
constexpr uint32_t MAXOWN = get_arg(args::MAXOWN);
constexpr uint32_t X_TRID0 = 8;
#ifdef DSMC_STREAMING
constexpr uint32_t X_SLOTS = 3;
#else
constexpr uint32_t X_SLOTS = 8;
#endif
static_assert(XL >= 1 && XL <= 8, "at most 8 activation reads in flight (trids 8..15)");
// NoC0 virtual coordinates (y << 16 | x) of the activation and output shard cores, in shard order.
constexpr uint32_t kXCores[] = {DSMC_XC};
constexpr uint32_t kOutCores[] = {DSMC_OC};
#ifdef DSMC_REDUCE
constexpr uint32_t PT = get_arg(args::PT);       // partial tile bytes (Dest precision)
constexpr uint32_t kGroupCores[] = {DSMC_CORE};  // per work item
#endif
#endif

constexpr UnicastEndpoint kCore;  // a worker core's L1 at virtual (x, y)

FORCE_INLINE uint32_t core_x(uint32_t xy) { return xy & 0xffff; }
FORCE_INLINE uint32_t core_y(uint32_t xy) { return xy >> 16; }

struct Ring {
    uint32_t base = 0, issued = 0, pushed = 0;
};
struct XQueue {
    uint32_t issued = 0, pushed = 0;
};

// A zero transaction count is not completion until the read command was accepted.
// Poll command readiness at retirement instead of waiting after each issue: the
// RISC can prepare more work while the command enters the NoC. Noc does not yet
// expose this readiness check; read_cmd_buf is the buffer used by its read API.
FORCE_INLINE bool read_batch_complete(const Noc& noc, uint32_t trid) {
    return noc_cmd_buf_ready(noc.get_noc_id(), read_cmd_buf) && noc.is_read_trid_flushed(trid);
}

FORCE_INLINE void wait_read_batch(const Noc& noc, uint32_t trid) {
    while (!noc_cmd_buf_ready(noc.get_noc_id(), read_cmd_buf)) {
    }
    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = trid});
}

#ifdef DSMC_STREAM1
// Push every activation block whose read has landed, in order.
FORCE_INLINE void retire_x(const Noc& noc, DataflowBuffer& dfb_x, XQueue& xq) {
    while (xq.pushed < xq.issued && read_batch_complete(noc, X_TRID0 + (xq.pushed % X_SLOTS))) {
        dfb_x.push_back(MT * KB);
        xq.pushed++;
    }
}

// Activation rows [k, k + n) into L1 at l1, tagged trid: runs that sit in one shard are one read.
FORCE_INLINE void read_x(const Noc& noc, uint32_t x_addr, uint32_t k, uint32_t n, uint32_t l1, uint32_t trid) {
    const uint32_t kend = k + n;
    while (k < kend) {
        const uint32_t s = k / SW, o = k % SW;
        const uint32_t run = (SW - o) < (kend - k) ? (SW - o) : (kend - k);
        noc.async_read<NocOptions::TXN_ID>(
            kCore,
            CoreLocalMem<uint32_t>(l1),
            run * XT,
            {.noc_x = core_x(kXCores[s]), .noc_y = core_y(kXCores[s]), .addr = x_addr + o * XT},
            {},
            {.trid = trid});
        l1 += run * XT;
        k += run;
    }
}
// A whole storage shard is contiguous across M, so issue one transfer for it.
// Partial-shard blocks retain row strides without an intermediate layout conversion.
FORCE_INLINE void read_x_block(const Noc& noc, uint32_t x_addr, uint32_t k, uint32_t rb, uint32_t l1, uint32_t trid) {
    if constexpr (MT > 1 && KB == SW) {
        if (rb == SW && k % SW == 0) {
            const uint32_t xy = kXCores[k / SW];
            noc.async_read<NocOptions::TXN_ID>(
                kCore,
                CoreLocalMem<uint32_t>(l1),
                MT * SW * XT,
                {.noc_x = core_x(xy), .noc_y = core_y(xy), .addr = x_addr},
                {},
                {.trid = trid});
            return;
        }
    }
    for (uint32_t m = 0; m < MT; ++m) {
        read_x(noc, x_addr + m * SW * XT, k, rb, l1 + m * KB * XT, trid);
    }
}

#endif

#ifdef DSMC_STREAMS_W
// Push every weight block whose read has landed, in order.
FORCE_INLINE void retire_w(const Noc& noc, DataflowBuffer& dfb_w, Ring& ring) {
    while (ring.pushed < ring.issued && read_batch_complete(noc, (ring.pushed % D) + 1)) {
        dfb_w.push_back(BT);
        ring.pushed++;
    }
}

// One block (rb rows x nc columns of this bank's shard, from row r and column n0) into the next
// ring slot. The ring is exactly D slots of the buffer, so slot i always lands where the buffer's
// write pointer will be when block i is pushed.
template <typename W, typename XRetire>
FORCE_INLINE void issue_w(
    const Noc& noc,
    DataflowBuffer& dfb_w,
    Ring& ring,
    XRetire&& retire_x_fn,
    const W& w,
    uint32_t bank,
    uint32_t r,
    uint32_t rb,
    uint32_t n0,
    uint32_t nc) {
    retire_w(noc, dfb_w, ring);
    while (!dfb_w.pages_reservable_at_back((ring.issued - ring.pushed + 1) * BT)) {
        retire_x_fn();  // compute may be waiting for an activation block before it frees a slot
        if (ring.pushed < ring.issued) {
            wait_read_batch(noc, (ring.pushed % D) + 1);
            dfb_w.push_back(BT);
            ring.pushed++;
        }
    }
    if (ring.issued == 0) {
        ring.base = dfb_w.get_write_ptr();
    }
    const uint32_t slot = ring.issued % D;
    const CoreLocalMem<uint32_t> dst(ring.base + slot * BT * WT);
    const NocOptVals tag{.trid = slot + 1};
    if (nc == Nbt) {
        // whole rows of the shard are contiguous in the bank
        noc.async_read<NocOptions::TXN_ID>(
            w, dst, rb * Nbt * WT, {.bank_id = bank, .offset_bytes = r * Nbt * WT}, {}, tag);
    } else {
        for (uint32_t i = 0; i < rb; i++) {
            noc.async_read<NocOptions::TXN_ID>(
                w,
                dst,
                nc * WT,
                {.bank_id = bank, .offset_bytes = ((r + i) * Nbt + n0) * WT},
                {.offset_bytes = i * nc * WT},
                tag);
        }
    }
    ring.issued++;
}

FORCE_INLINE void drain_w(const Noc& noc, DataflowBuffer& dfb_w, Ring& ring) {
    while (ring.pushed < ring.issued) {
        wait_read_batch(noc, (ring.pushed % D) + 1);
        dfb_w.push_back(BT);
        ring.pushed++;
    }
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t role = kRole[get_absolute_logical_y() * GX + get_absolute_logical_x()];
    if (role == 0) {
        return;
    }
    const uint32_t wi = role - 1, kidx = wi % CK, grp = wi / CK, cn = grp % CN;
    const uint32_t cm = (grp / CN) % CM, bank = grp / (CM * CN);
    const uint32_t n0 = Nbt * cn / CN, n1 = Nbt * (cn + 1) / CN;
    const uint32_t k0 = KT * kidx / CK, k1 = KT * (kidx + 1) / CK;
    const uint32_t nc = n1 - n0, rows = k1 - k0, nblk = (rows + KB - 1) / KB;
    // Blocks start at a per-core rotated position so cores that read the same activation rows do
    // not hit the same shard at the same time.
    const uint32_t rot = ROT ? grp * nblk / NGRP : 0;
    const Noc noc;  // this kernel's NoC (stream 0: NOC_0, stream 1: NOC_1)

#ifdef DSMC_STREAMS_W
    DataflowBuffer dfb_w(dfb::w);
    const TensorAccessor w_acc(tensor::in1);
    const BankView w(w_acc);  // weight shard of DRAM bank `bank`, addressed by offset into it
    Ring ring;
#endif
#ifdef DSMC_STREAM1
    DataflowBuffer dfb_x(dfb::x);
    // Raw: same L1 address on the activation shard cores (no handle form yet).
    const uint32_t x_addr =
        tensor_accessor::detail::TransferAccess::bank_base(TensorAccessor(tensor::in0)) + cm * MT * SW * XT;
#ifndef DSMC_STREAMING
    dfb_x.reserve_back(nblk * MT * KB);
#endif
    const uint32_t xl1 = dfb_x.get_write_ptr();
    XQueue xq;
    [[maybe_unused]] auto retire_x_fn = [&]() { retire_x(noc, dfb_x, xq); };
#else
    [[maybe_unused]] auto retire_x_fn = []() {};
#endif

    for (uint32_t q = 0; q < P; q++) {
        [[maybe_unused]] const uint32_t qa = n0 + nc * q / P, qb = n0 + nc * (q + 1) / P;
        for (uint32_t p = 0; p < nblk; p++) {
            [[maybe_unused]] const uint32_t j = (p + rot) % nblk, r = j * KB, rb = (rows - r) < KB ? (rows - r) : KB;
#ifdef DSMC_STREAM1
            if (q == 0) {
                // Issue ahead without a per-block barrier. While waiting for a reusable
                // X slot, publish W completions too: compute needs both to return credits.
                while (xq.issued < nblk && xq.issued < p + XL) {
#ifdef DSMC_STREAMING
                    retire_x_fn();
                    if (!dfb_x.pages_reservable_at_back((xq.issued - xq.pushed + 1) * MT * KB) && xq.issued > p) {
                        break;
                    }
                    while (!dfb_x.pages_reservable_at_back((xq.issued - xq.pushed + 1) * MT * KB)) {
                        retire_x_fn();
#ifdef DSMC_STREAMS_W
                        retire_w(noc, dfb_w, ring);
#endif
                    }
                    const uint32_t slot = xq.issued % X_SLOTS;
#else
                    if (xq.issued - xq.pushed >= X_SLOTS) {
                        wait_read_batch(noc, X_TRID0 + (xq.pushed % X_SLOTS));
                        dfb_x.push_back(MT * KB);
                        xq.pushed++;
                    }
                    const uint32_t slot = xq.issued;
#endif
                    const uint32_t jx = (xq.issued + rot) % nblk, rx = jx * KB;
                    const uint32_t rbx = (rows - rx) < KB ? (rows - rx) : KB;
                    read_x_block(noc, x_addr, k0 + rx, rbx, xl1 + slot * MT * KB * XT, X_TRID0 + (xq.issued % X_SLOTS));
                    xq.issued++;
#ifdef DSMC_X_SYNC
                    wait_read_batch(noc, X_TRID0 + ((xq.issued - 1) % X_SLOTS));
                    dfb_x.push_back(MT * KB);
                    xq.pushed++;
#endif
                }
                retire_x_fn();
            }
#endif
#ifdef DSMC_STREAMS_W
            if (WSINGLE ? (PAR == 0) : ((p & 1) == PAR)) {
                issue_w(noc, dfb_w, ring, retire_x_fn, w, bank, k0 + r, rb, qa, qb - qa);
            } else {
                retire_w(noc, dfb_w, ring);
            }
#endif
        }
#ifdef DSMC_STREAM1
        if (q == 0) {
            while (xq.pushed < nblk) {
                wait_read_batch(noc, X_TRID0 + (xq.pushed % X_SLOTS));
                dfb_x.push_back(MT * KB);
                xq.pushed++;
            }
        }
#endif
    }
#ifdef DSMC_STREAMS_W
    drain_w(noc, dfb_w, ring);
#endif
    // The read trid is sticky command-buffer state that Noc::async_read<TXN_ID> sets per call; the
    // Noc class has no call that clears it, so put back the default tag (0) the legacy way, as
    // indexed_fused_update_cache.cpp does, rather than leak trid 1..15 to the next kernel on this RISC.
    noc_async_read_set_trid(0, noc.get_noc_id());

#ifdef DSMC_STREAM1
    const uint32_t o0 = n0 + nc * kidx / CK, o1 = n0 + nc * (kidx + 1) / CK, nown = o1 - o0;
#ifdef DSMC_REDUCE
    {
        // Reduce-scatter inside the column group: member q owns columns [n0 + nc*q/CK, n0 + nc*(q+1)/CK).
        // Every member writes the slice of its partials that member q owns into slot kidx of q's
        // receive buffer (its own slice to itself), then bumps q's semaphore.
        DataflowBuffer dfb_part(dfb::part);
        DataflowBuffer dfb_recv(dfb::recv);
        Semaphore sem(sem::reduce);
        const uint32_t recv = dfb_recv.get_write_ptr();
        dfb_part.wait_front(MT * NCP);
        const uint32_t part = dfb_part.get_read_ptr();
        for (uint32_t q = 0; q < CK; q++) {
            const uint32_t pxy = kGroupCores[grp * CK + q];
            const uint32_t pc0 = n0 + nc * q / CK, pc1 = n0 + nc * (q + 1) / CK;
            if (pc1 > pc0) {
                for (uint32_t m = 0; m < MT; ++m) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(part),
                        kCore,
                        (pc1 - pc0) * PT,
                        {.offset_bytes = (m * NCP + pc0 - n0) * PT},
                        {.noc_x = core_x(pxy), .noc_y = core_y(pxy), .addr = recv + (kidx * MT + m) * MAXOWN * PT});
                }
            }
        }
        noc.async_write_barrier();
        for (uint32_t q = 0; q < CK; q++) {
            if (q != kidx) {
                const uint32_t pxy = kGroupCores[grp * CK + q];
                sem.up(noc, core_x(pxy), core_y(pxy), 1);
            }
        }
        dfb_part.pop_front(MT * NCP);
        dfb_recv.reserve_back(CK * MT * MAXOWN);
        sem.wait(CK - 1);
        sem.set(0);
        dfb_recv.push_back(CK * MT * MAXOWN);
    }
#endif
    {
        DataflowBuffer dfb_out(dfb::out);
        // Raw: same L1 address on the output shard cores (no handle form yet).
        const uint32_t out_addr = tensor_accessor::detail::TransferAccess::bank_base(TensorAccessor(tensor::output));
        dfb_out.wait_front(MT * MAXOWN);
        const uint32_t l1 = dfb_out.get_read_ptr();
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t i = 0; i < nown;) {
                const uint32_t n = bank * Nbt + o0 + i;
                if (n >= OCAP) {
                    break;
                }
                const uint32_t offset = n % OW;
                const uint32_t run = (OW - offset) < (nown - i) ? OW - offset : nown - i;
                const uint32_t oxy = kOutCores[n / OW];
                noc.async_write(
                    CoreLocalMem<uint32_t>(l1),
                    kCore,
                    run * OT,
                    {.offset_bytes = (m * MAXOWN + i) * OT},
                    {.noc_x = core_x(oxy),
                     .noc_y = core_y(oxy),
                     .addr = out_addr + ((cm * MT + m) * OW + offset) * OT});
                i += run;
            }
        }
        noc.async_write_barrier();
        dfb_out.pop_front(MT * MAXOWN);
    }
#ifdef DSMC_REDUCE
    noc.async_atomic_barrier();
#endif
#endif
}
