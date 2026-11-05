// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data movement for the multi-row DRAM-streamed matmul (MatmulMultiCoreReuseMultiCast1DProgramConfig with
// dram_stream_workers_per_bank > 0): an activation of MT tile rows (128 rows at MT = 4) times a DRAM width-sharded
// weight. The program runs on the whole worker grid; DSMC_ROLE gives each logical core a role:
//   0              idle: tells the distributors it runs this program, then exits;
//   1 .. NW        work item wi = role - 1 = bank * CN + cn: owns weight columns [n0, n1) of `bank`'s shard and
//                  streams them from that bank over the whole K (the weight-streaming layout of the multi-core
//                  DRAM-sharded decode pipeline, with its cores placed next to their bank);
//   NW + 1 + d     activation distributor d: reads K chunks g = d, d + NX, ... of the activation from its interleaved
//                  tensor into its own copy of dfb::x and multicasts each chunk into dfb::x of every core of the grid,
//                  then multicasts its chunk count into semaphore xs[d].
// The activation lands once, resident, k-major (tile (m, k) at k * MT + m) so that every K chunk is one contiguous
// range of dfb::x that a work core pushes as soon as its distributor's count passes it. Before the first multicast a
// distributor waits until every other core of the grid has signalled `ready`, so no multicast lands in a core still
// running the previous program.
//
// The same source is built twice:
//   stream 0 (NOC_0): streams weight blocks into dfb::w (even blocks; every block when WSINGLE).
//   stream 1 (NOC_1, MR_STREAM1): the ready signal, the distributor role, pushes landed activation chunks, streams
//     odd weight blocks, and writes the output tiles into the interleaved output.

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t Nbt = get_arg(args::Nbt);          // weight shard width per bank (tiles)
constexpr uint32_t KT = get_arg(args::KT);            // K (tiles)
constexpr uint32_t KB = get_arg(args::KB);            // K rows per streamed weight block
constexpr uint32_t CN = get_arg(args::CN);            // work cores per bank
constexpr uint32_t P = get_arg(args::P);              // column passes per core
constexpr uint32_t NCP = get_arg(args::NCP);          // columns per pass slot (weight block width in L1)
constexpr uint32_t D = get_arg(args::D);              // weight blocks in flight per stream (trids 1..D)
constexpr uint32_t WT = get_arg(args::WT);            // weight tile bytes (DRAM-aligned)
constexpr uint32_t GX = get_arg(args::GX);            // worker grid width (DSMC_ROLE row stride)
constexpr uint32_t PAR = get_arg(args::PAR);          // 0: stream 0, 1: stream 1
constexpr uint32_t WSINGLE = get_arg(args::WSINGLE);  // 1: stream 0 reads every block
constexpr uint32_t NW = get_arg(args::NW);            // work items
constexpr uint32_t BT = KB * NCP;                     // dfb::w entries per block slot
constexpr uint32_t NBLK = (KT + KB - 1) / KB;
static_assert(D >= 1 && D <= 7, "weight trids 1..D");

constexpr uint8_t kRole[] = {DSMC_ROLE};

#ifndef MR_LINKED
#define MR_LINKED true
#endif
#ifdef MR_STREAM1
constexpr uint32_t MT = get_arg(args::MT);      // activation / output tile rows
constexpr uint32_t XT = get_arg(args::XT);      // activation tile bytes
constexpr uint32_t NX = get_arg(args::NX);      // distributors (<= 4)
constexpr uint32_t XC = get_arg(args::XC);      // K tiles per activation chunk
constexpr uint32_t OT = get_arg(args::OT);      // output tile bytes
constexpr uint32_t OD = get_arg(args::OD);      // 2 with fuse_swiglu: one output tile per gate / up pair
constexpr uint32_t NOUT = get_arg(args::NOUT);  // output width (tiles)
constexpr uint32_t MX0 = get_arg(args::MX0);    // multicast rectangle (NOC_1 order), the whole grid
constexpr uint32_t MY0 = get_arg(args::MY0);
constexpr uint32_t MX1 = get_arg(args::MX1);
constexpr uint32_t MY1 = get_arg(args::MY1);
constexpr uint32_t MNUM = get_arg(args::MNUM);  // grid cores but the sender
constexpr uint32_t NCH = (KT + XC - 1) / XC;
static_assert(NX >= 1 && NX <= 3, "1..3 activation distributors (xs3 is the multicast turn token)");
constexpr uint32_t kDist[] = {MR_DIST};  // NoC virtual (y << 16 | x) of the distributors
#endif

constexpr AllocatorBank<AllocatorBankType::DRAM> kDram;

FORCE_INLINE uint32_t core_x(uint32_t xy) { return xy & 0xffff; }
FORCE_INLINE uint32_t core_y(uint32_t xy) { return xy >> 16; }

struct Ring {
    uint32_t base = 0, issued = 0, pushed = 0;
};

// Push every weight block whose read has landed, in order.
FORCE_INLINE void retire_w(const Noc& noc, DataflowBuffer& dfb_w, Ring& ring) {
    while (ring.pushed < ring.issued && noc.is_read_trid_flushed((ring.pushed % D) + 1)) {
        dfb_w.push_back(BT);
        ring.pushed++;
    }
}

// One block (rb rows x nc columns of this bank's shard, from row r and column n0) into the next ring slot. The ring
// is exactly D slots of the buffer, so slot i always lands where the buffer's write pointer is when block i is pushed.
template <typename Poll>
FORCE_INLINE void issue_w(
    const Noc& noc,
    DataflowBuffer& dfb_w,
    Ring& ring,
    Poll&& poll,
    uint32_t bank,
    uint32_t w_addr,
    uint32_t r,
    uint32_t rb,
    uint32_t n0,
    uint32_t nc) {
    retire_w(noc, dfb_w, ring);
    while (!dfb_w.pages_reservable_at_back((ring.issued - ring.pushed + 1) * BT)) {
        poll();  // compute may be waiting for an activation chunk before it frees a slot
        if (ring.pushed < ring.issued) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = (ring.pushed % D) + 1});
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
#ifdef MR_ABL_NO_W
    if (false) {
#else
    if (nc == Nbt) {
#endif
        noc.async_read<NocOptions::TXN_ID>(
            kDram, dst, rb * Nbt * WT, {.bank_id = bank, .addr = w_addr + r * Nbt * WT}, {}, tag);
    } else {
#ifndef MR_ABL_NO_W
        for (uint32_t i = 0; i < rb; i++) {
            noc.async_read<NocOptions::TXN_ID>(
                kDram,
                dst,
                nc * WT,
                {.bank_id = bank, .addr = w_addr + ((r + i) * Nbt + n0) * WT},
                {.offset_bytes = i * nc * WT},
                tag);
        }
#endif
    }
    ring.issued++;
}

FORCE_INLINE void drain_w(const Noc& noc, DataflowBuffer& dfb_w, Ring& ring) {
    while (ring.pushed < ring.issued) {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = (ring.pushed % D) + 1});
        dfb_w.push_back(BT);
        ring.pushed++;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t role = kRole[get_absolute_logical_y() * GX + get_absolute_logical_x()];
    const Noc noc;  // stream 0: NOC_0, stream 1: NOC_1

#ifdef MR_STREAM1
    Semaphore<> ready(sem::ready);
    Semaphore<> xs[4] = {Semaphore<>(sem::xs0), Semaphore<>(sem::xs1), Semaphore<>(sem::xs2), Semaphore<>(sem::xs3)};
    DataflowBuffer dfb_x(dfb::x);
    const uint32_t xb = dfb_x.get_write_ptr();
    {
        // Every core of the grid but a distributor itself tells each distributor that it runs this program.
        const uint32_t self = role > NW ? role - NW - 1 : NX;
        for (uint32_t d = 0; d < NX; d++) {
            if (d != self) {
                ready.up(noc, core_x(kDist[d]), core_y(kDist[d]), 1);
            }
        }
    }
    if (role == 0) {
        noc.async_atomic_barrier();
        return;
    }
    if (role > NW) {
        // Activation distributor d.
        const uint32_t d = role - NW - 1;
        const auto in0 = TensorAccessor(tensor::in0);
        ready.wait(MNUM);
        ready.set(0);
        uint32_t count = 0;
        Semaphore<>& turn = xs[3];
        for (uint32_t g = d; g < NCH; g += NX) {
            const uint32_t k0 = g * XC, kn = (KT - k0) < XC ? (KT - k0) : XC;
            const uint32_t base = xb + k0 * MT * XT;
#ifndef MR_ABL_NO_X
            for (uint32_t k = 0; k < kn; k++) {
                for (uint32_t m = 0; m < MT; m++) {
                    noc.async_read(
                        in0, CoreLocalMem<uint32_t>(base + (k * MT + m) * XT), XT, {.page_id = m * KT + k0 + k}, {});
                }
            }
            noc.async_read_barrier();
            if (NX > 1 && g > 0) {
                turn.wait_min(count + (d == 0 ? 0 : 1));  // chunk g - 1's multicast has gone out
            }
            const MulticastEndpoint mcast_dst;
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(base),
                mcast_dst,
                kn * MT * XT,
                MNUM,
                {},
                {.noc_x_start = MX0, .noc_y_start = MY0, .noc_x_end = MX1, .noc_y_end = MY1, .addr = base},
                MR_LINKED);
#endif
            noc.async_writes_flushed();
            count++;
            xs[d].set(count);
            xs[d].set_multicast(noc, MX0, MY0, MX1, MY1, MNUM);
            if (NX > 1 && g + 1 < NCH) {
                const uint32_t next = kDist[(g + 1) % NX];
                turn.up(noc, core_x(next), core_y(next), 1);
            }
        }
        noc.async_write_barrier();
        noc.async_atomic_barrier();
        xs[d].set(0);
        turn.set(0);
        return;
    }
#else
    if (role == 0 || role > NW) {
        return;
    }
#endif

    const uint32_t wi = role - 1, cn = wi % CN, bank = wi / CN;
    const uint32_t n0 = Nbt * cn / CN, n1 = Nbt * (cn + 1) / CN, nc = n1 - n0;
#ifdef MR_STREAMS_W
    DataflowBuffer dfb_w(dfb::w);
    const uint32_t w_addr = TensorAccessor(tensor::in1).get_bank_base_address();
    Ring ring;
#endif

#ifdef MR_STREAM1
    dfb_x.reserve_back(KT * MT);
    uint32_t xg = 0;  // next activation chunk to push
    auto poll = [&]() {
        while (xg < NCH) {
            if (xs[xg % NX].value() < xg / NX + 1) {
                break;
            }
            const uint32_t kn = (KT - xg * XC) < XC ? (KT - xg * XC) : XC;
            dfb_x.push_back(kn * MT);
            xg++;
        }
    };
#else
    auto poll = []() {};
#endif

    for (uint32_t q = 0; q < P; q++) {
        const uint32_t qa = n0 + nc * q / P, qb = n0 + nc * (q + 1) / P;
        for (uint32_t p = 0; p < NBLK; p++) {
            const uint32_t r = p * KB, rb = (KT - r) < KB ? (KT - r) : KB;
            poll();
#ifdef MR_STREAMS_W
            if (WSINGLE ? (PAR == 0) : ((p & 1) == PAR)) {
                issue_w(noc, dfb_w, ring, poll, bank, w_addr, r, rb, qa, qb - qa);
            } else {
                retire_w(noc, dfb_w, ring);
            }
#else
            (void)r;
            (void)rb;
            (void)qa;
            (void)qb;
#endif
        }
    }
#ifdef MR_STREAMS_W
    drain_w(noc, dfb_w, ring);
#endif
    // The read trid is sticky command-buffer state that Noc::async_read<TXN_ID> sets per call; put back the default.
    noc_async_read_set_trid(0, noc.get_noc_id());

#ifdef MR_STREAM1
    while (xg < NCH) {
        poll();
    }
    {
        DataflowBuffer dfb_out(dfb::out);
        const auto out = TensorAccessor(tensor::output);
        for (uint32_t q = 0; q < P; q++) {
            const uint32_t qa = nc * q / P, nq = (nc * (q + 1) / P - qa) / OD;
            dfb_out.wait_front(MT * nq);
            const uint32_t l1 = dfb_out.get_read_ptr();
            const uint32_t col0 = (bank * Nbt + n0 + qa) / OD;
            for (uint32_t m = 0; m < MT; m++) {
                for (uint32_t j = 0; j < nq; j++) {
                    if (col0 + j < NOUT) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(l1 + (m * nq + j) * OT),
                            out,
                            OT,
                            {},
                            {.page_id = m * NOUT + col0 + j});
                    }
                }
            }
            noc.async_write_barrier();
            dfb_out.pop_front(MT * nq);
        }
    }
    noc.async_atomic_barrier();
#endif
}
