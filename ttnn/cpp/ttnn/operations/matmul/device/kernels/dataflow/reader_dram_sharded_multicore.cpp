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
// Work item wi = (bank * CN + cn) * CK + kidx owns weight columns [n0, n1) of `bank`'s shard and
// K rows [k0, k1). Every range below is derived from wi with the factory's formulas, so no per-core
// runtime arguments are needed; the only per-core table is DSMC_ROLE (one byte per logical core:
// 0 = idle, else wi + 1), which lets the program run on the bounding rectangle of its cores.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#ifdef DSMC_REDUCE
#include "api/semaphore.h"
#endif

namespace {

constexpr uint32_t Nbt = get_arg(args::Nbt);    // weight shard width per bank (tiles)
constexpr uint32_t KT = get_arg(args::KT);      // K (tiles)
constexpr uint32_t KB = get_arg(args::KB);      // K rows per streamed block
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
static_assert(XL >= 1 && XL <= 8, "at most 8 activation reads in flight (trids 8..15)");
// NoC0 virtual coordinates (y << 16 | x) of the activation and output shard cores, in shard order.
constexpr uint32_t kXCores[] = {DSMC_XC};
constexpr uint32_t kOutCores[] = {DSMC_OC};
#ifdef DSMC_REDUCE
constexpr uint32_t PT = get_arg(args::PT);       // partial tile bytes (Dest precision)
constexpr uint32_t kGroupCores[] = {DSMC_CORE};  // per work item
#endif
#endif

FORCE_INLINE uint64_t core_noc_addr(uint32_t xy, uint32_t addr) { return get_noc_addr(xy & 0xffff, xy >> 16, addr); }

struct Ring {
    uint32_t base = 0, issued = 0, pushed = 0;
};
struct XQueue {
    uint32_t issued = 0, pushed = 0;
};

#ifdef DSMC_STREAM1
// Push every activation block whose read has landed, in order.
FORCE_INLINE void retire_x(DataflowBuffer& dfb_x, XQueue& xq) {
    while (xq.pushed < xq.issued &&
           ncrisc_noc_read_with_transaction_id_flushed(noc_index, X_TRID0 + (xq.pushed & 7))) {
        dfb_x.push_back(KB);
        xq.pushed++;
    }
}

// Activation rows [k, k + n) into L1 at l1: runs that sit in one shard are one read.
FORCE_INLINE void read_x(uint32_t x_addr, uint32_t k, uint32_t n, uint32_t l1) {
    const uint32_t kend = k + n;
    while (k < kend) {
        const uint32_t s = k / SW, o = k % SW;
        const uint32_t run = (SW - o) < (kend - k) ? (SW - o) : (kend - k);
        noc_async_read(core_noc_addr(kXCores[s], x_addr + o * XT), l1, run * XT);
        l1 += run * XT;
        k += run;
    }
}
#endif

#ifdef DSMC_STREAMS_W
// Push every weight block whose read has landed, in order.
FORCE_INLINE void retire_w(DataflowBuffer& dfb_w, Ring& ring) {
    while (ring.pushed < ring.issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, (ring.pushed % D) + 1)) {
        dfb_w.push_back(BT);
        ring.pushed++;
    }
}

// One block (rb rows x nc columns of this bank's shard, from row r and column n0) into the next
// ring slot. The ring is exactly D slots of the buffer, so slot i always lands where the buffer's
// write pointer will be when block i is pushed.
template <typename XRetire>
FORCE_INLINE void issue_w(
    DataflowBuffer& dfb_w,
    Ring& ring,
    XRetire&& retire_x_fn,
    uint32_t bank,
    uint32_t w_addr,
    uint32_t r,
    uint32_t rb,
    uint32_t n0,
    uint32_t nc) {
    retire_w(dfb_w, ring);
    while (!dfb_w.pages_reservable_at_back((ring.issued - ring.pushed + 1) * BT)) {
        retire_x_fn();  // compute may be waiting for an activation block before it frees a slot
        if (ring.pushed < ring.issued) {
            noc_async_read_barrier_with_trid((ring.pushed % D) + 1);
            dfb_w.push_back(BT);
            ring.pushed++;
        }
    }
    if (ring.issued == 0) {
        ring.base = dfb_w.get_write_ptr();
    }
    const uint32_t slot = ring.issued % D;
    const uint32_t dst = ring.base + slot * BT * WT;
    noc_async_read_set_trid(slot + 1);
    if (nc == Nbt) {
        // whole rows of the shard are contiguous in the bank
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_addr + r * Nbt * WT), dst, rb * Nbt * WT);
    } else {
        for (uint32_t i = 0; i < rb; i++) {
            noc_async_read(
                get_noc_addr_from_bank_id<true>(bank, w_addr + ((r + i) * Nbt + n0) * WT), dst + i * nc * WT, nc * WT);
        }
    }
    ring.issued++;
}

FORCE_INLINE void drain_w(DataflowBuffer& dfb_w, Ring& ring) {
    while (ring.pushed < ring.issued) {
        noc_async_read_barrier_with_trid((ring.pushed % D) + 1);
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
    const uint32_t wi = role - 1, kidx = wi % CK, grp = wi / CK, cn = grp % CN, bank = grp / CN;
    const uint32_t n0 = Nbt * cn / CN, n1 = Nbt * (cn + 1) / CN;
    const uint32_t k0 = KT * kidx / CK, k1 = KT * (kidx + 1) / CK;
    const uint32_t nc = n1 - n0, rows = k1 - k0, nblk = (rows + KB - 1) / KB;
    // Blocks start at a per-core rotated position so cores that read the same activation rows do
    // not hit the same shard at the same time.
    const uint32_t rot = ROT ? grp * nblk / NGRP : 0;

#ifdef DSMC_STREAMS_W
    DataflowBuffer dfb_w(dfb::w);
    const uint32_t w_addr = TensorAccessor(tensor::in1).get_bank_base_address();
    Ring ring;
#endif
#ifdef DSMC_STREAM1
    DataflowBuffer dfb_x(dfb::x);
    const uint32_t x_addr = TensorAccessor(tensor::in0).get_bank_base_address();
    dfb_x.reserve_back(nblk * KB);
    const uint32_t xl1 = dfb_x.get_write_ptr();
    XQueue xq;
    [[maybe_unused]] auto retire_x_fn = [&]() { retire_x(dfb_x, xq); };
#else
    [[maybe_unused]] auto retire_x_fn = []() {};
#endif

    for (uint32_t q = 0; q < P; q++) {
        [[maybe_unused]] const uint32_t qa = n0 + nc * q / P, qb = n0 + nc * (q + 1) / P;
        for (uint32_t p = 0; p < nblk; p++) {
            [[maybe_unused]] const uint32_t j = (p + rot) % nblk, r = j * KB, rb = (rows - r) < KB ? (rows - r) : KB;
#ifdef DSMC_STREAM1
            if (q == 0) {
                // the activation is read once, in stream order, XL blocks ahead
                while (xq.issued < nblk && xq.issued < p + XL) {
                    if (xq.issued - xq.pushed >= 8) {
                        noc_async_read_barrier_with_trid(X_TRID0 + (xq.pushed & 7));
                        dfb_x.push_back(KB);
                        xq.pushed++;
                    }
                    const uint32_t jx = (xq.issued + rot) % nblk, rx = jx * KB;
                    const uint32_t rbx = (rows - rx) < KB ? (rows - rx) : KB;
                    noc_async_read_set_trid(X_TRID0 + (xq.issued & 7));
                    read_x(x_addr, k0 + rx, rbx, xl1 + xq.issued * KB * XT);
                    xq.issued++;
                }
                retire_x(dfb_x, xq);
            }
#endif
#ifdef DSMC_STREAMS_W
            if (WSINGLE ? (PAR == 0) : ((p & 1) == PAR)) {
                issue_w(dfb_w, ring, retire_x_fn, bank, w_addr, k0 + r, rb, qa, qb - qa);
            } else {
                retire_w(dfb_w, ring);
            }
#endif
        }
#ifdef DSMC_STREAM1
        if (q == 0) {
            while (xq.pushed < nblk) {
                noc_async_read_barrier_with_trid(X_TRID0 + (xq.pushed & 7));
                dfb_x.push_back(KB);
                xq.pushed++;
            }
        }
#endif
    }
#ifdef DSMC_STREAMS_W
    drain_w(dfb_w, ring);
#endif
    noc_async_read_set_trid(0);

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
        const Noc noc;
        const uint32_t recv = dfb_recv.get_write_ptr();
        dfb_part.wait_front(NCP);
        const uint32_t part = dfb_part.get_read_ptr();
        for (uint32_t q = 0; q < CK; q++) {
            const uint32_t pxy = kGroupCores[grp * CK + q];
            const uint32_t pc0 = n0 + nc * q / CK, pc1 = n0 + nc * (q + 1) / CK;
            if (pc1 > pc0) {
                noc_async_write(
                    part + (pc0 - n0) * PT, core_noc_addr(pxy, recv + kidx * MAXOWN * PT), (pc1 - pc0) * PT);
            }
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < CK; q++) {
            if (q != kidx) {
                const uint32_t pxy = kGroupCores[grp * CK + q];
                sem.up(noc, pxy & 0xffff, pxy >> 16, 1);
            }
        }
        dfb_part.pop_front(NCP);
        dfb_recv.reserve_back(CK * MAXOWN);
        sem.wait(CK - 1);
        sem.set(0);
        dfb_recv.push_back(CK * MAXOWN);
    }
#endif
    {
        DataflowBuffer dfb_out(dfb::out);
        const uint32_t out_addr = TensorAccessor(tensor::output).get_bank_base_address();
        dfb_out.wait_front(MAXOWN);
        const uint32_t l1 = dfb_out.get_read_ptr();
        for (uint32_t i = 0; i < nown; i++) {
            const uint32_t n = bank * Nbt + o0 + i;  // global output column (tile)
            if (n < OCAP) {
                noc_async_write(l1 + i * OT, core_noc_addr(kOutCores[n / OW], out_addr + (n % OW) * OT), OT);
            }
        }
        noc_async_write_barrier();
        dfb_out.pop_front(MAXOWN);
    }
#ifdef DSMC_REDUCE
    noc_async_atomic_barrier();
#endif
#endif
}
