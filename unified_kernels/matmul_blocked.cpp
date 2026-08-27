// SPDX-License-Identifier: Apache-2.0
//
// A 2D-BLOCKED MATMUL: [M, K] @ [K, N], for the shapes that do not fit L1 whole.
//
// This started as the attention output projection and generalised into the thing that
// projection turned out to be. It now drives all four of a layer's large matmuls, which
// differ only in extents: the output projection is K = N = d_model, the gate and up
// projections are K = d_model and N = ffn, and the down projection is the reverse.
//
// matmul.cpp remains the single-shot kernel for blocks that DO fit: it holds each operand
// whole and has no blocking loops, which is cheaper when it is possible.
//
// BOTH dimensions are blocked, and that is the point rather than blocking K harder. B is
// ktot*ntot tiles: 64 at 256x256, but 4096 -- 8MB -- for a 2048x2048 projection and 16384 for
// a 2048x8192 FFN matrix, all far past L1. So the output is walked in [mt, nt] blocks and K in
// kb blocks of kt tiles, and EVERY operand is gathered by a custom load, because none of the
// three slices is contiguous in its backing tensor once both dimensions are cut. That is free:
// one read per page is what a contiguous block load already issues, so only the addresses
// differ. Gathering B also normalises its row stride to nt, so the matmul geometry never has
// to know it came out of a wider matrix.
//
// Over mtot total row-tiles the DRAM traffic is
//
//     mtot * ktot * ntot * (1/mt + 1/nt)      tiles
//
// -- every M-block reads all of B, every output-column block reads all of A -- subject to the
// output block and its partial both fitting L1, i.e. 2*mt*nt tiles plus operands. Taking
// nt == ntot makes the first term dominate and forces mt small, which is the trap; balancing
// the two is what wins. For [512, 2048] @ [2048, 2048] the model puts mt=4/nt=64 at 17408
// tiles and mt=8/nt=16 at 12288, and they measure 4531.6us and 3711.0us against ttnn.matmul's
// 3962.9us on one core.
//
// kt == ktot gives kb == 1, and that skips the accumulator entirely: with one k-block there is
// no partial to carry, so paying a pack and a reload for it would be waste.
//
// What each thread ends up executing, per query chunk:
//
//   NCRISC   per k-block: cb_a (mt x kt) and cb_b (kt x nt), both gathered
//   TRISC    matmul per k-block into the accumulator, then the mt x nt block in subblocks
//   BRISC    drain cb_out (mt * nt tiles) per output block
//
// Compile-time args, all named, plus a cb_<name> per buffer:
//   mt          rows per M-block, in tiles
//   ktot        K in tiles
//   ntot        N in tiles
//   kt          tiles per k-block; kt == ktot means no k-loop
//   nt          tiles per output-column block; nt == ntot means the full width
//
// Defines:
//   MMB_ACC_DST                 carry the partial in DST instead of L1 (slower here)
//   MMB_MCAST                   broadcast each operand to the cores that share it, which
//                               needs MMB_GRID_H x MMB_GRID_W == mb x nb exactly
//   MMB_GRID_H / MMB_GRID_W     the core grid, MMB_MCAST only
//   MMB_IN0_THREAD / MMB_IN1_THREAD  which DM thread, and so which NOC, carries each
//                               operand. NOT symmetric: NOC 0 is measurably better for
//                               these DRAM reads, so the BIG operand belongs on it. See
//                               unified_llama_prefill.md.
//            that the scalars above are named rather than positional
//
//   MMB_SHARE_PAIR              TEST ONLY. Put BOTH collectives on handshake pair 0
//                               instead of 0 and 1 -- the thing the pair parameter exists
//                               to prevent. Whether it actually breaks is what MMB_SKEW is
//                               for; see unified_api_hazards.md hazard 13b.
//   MMB_SKEW                    TEST ONLY. Busy-wait this many iterations on row 0's
//                               RECEIVERS before each row load, so the row-0 sender parks
//                               in its ready wait while lower rows race ahead into their
//                               column collective. That is the exact interleaving the
//                               shared-pair hazard needs: a core incrementing (0,0)'s
//                               ready counter for the COLUMN collective while (0,0) is
//                               still counting ROW receivers.
//
// Runtime args, named and identical on all three kernels:
//   A base address, an [M, K] tensor
//   B base address, a [K, N] tensor
//   out base address, an [M, N] tensor
//   first output BLOCK this core owns, as a flat (m, n) index
//   how many output blocks this core owns

// A DM thread is bound to a NOC by its index, so choosing the thread chooses the NOC, and
// the two are NOT interchangeable for DRAM reads. B is the big operand and wants NOC 0; A
// then wants the other one so the two streams overlap. See unified_llama_prefill.md.
#ifndef MMB_IN0_THREAD
#define MMB_IN0_THREAD 1
#endif
#ifndef MMB_IN1_THREAD
#define MMB_IN1_THREAD 0
#endif

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t mt = get_named_compile_time_arg_val("mt");
    constexpr uint32_t ktot = get_named_compile_time_arg_val("ktot");
    constexpr uint32_t ntot = get_named_compile_time_arg_val("ntot");
    constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");

    constexpr uint32_t kCbIn = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t kCbWo = get_named_compile_time_arg_val("cb_wo");
    constexpr uint32_t kCbOut = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t kCbAcc = get_named_compile_time_arg_val("cb_acc");

    static_assert(kt > 0 && ktot % kt == 0, "the k-block width must divide K");
    static_assert(nt > 0 && ntot % nt == 0, "the output-column block width must divide N");
    constexpr uint32_t kb = ktot / kt;
    constexpr uint32_t nb = ntot / nt;
    // The unit of work across cores is one OUTPUT BLOCK -- an (m, n) tile -- indexed flat as
    // m*nb + n. Both dimensions are split, and neither needs a reduction: two cores holding
    // different m or different n write disjoint parts of the output. Only K would need one,
    // which is why K stays inside a core.
    //
    // Handing out M-blocks alone, as this did, made the blocking fight the parallelism: mt
    // has to be LARGE for traffic (it goes as 1/mt) and SMALL to make enough blocks to
    // spread. At [512,2048]@[2048,2048] that meant 16 cores at mt=1 scored 3371.7us against
    // 3710.7us for ONE core at mt=8 -- the extra cores bought almost nothing. Splitting the
    // output both ways gives mb*nb units instead of mb, so a large mt and a high core count
    // stop being alternatives.
    const uint32_t block_begin = get_arg(args::block_begin);
    const uint32_t block_count = get_arg(args::block_count);

    using A = u::Shape<mt, kt>;    // one (m, k) tile of A
    using W = u::Shape<kt, nt>;    // one (k, n) tile of B
    using Out = u::Shape<mt, nt>;  // one (m, n) tile of the output

    u::matmul_init<A, W>(kCbIn, kCbWo, kCbOut);

    u::Storage<A> a_storage(kCbIn);
    u::Storage<W> w_storage(kCbWo);
    u::Storage<Out> acc_storage(kCbAcc);
    u::Storage<Out> out_storage(kCbOut);

    const auto a_acc = TensorAccessor(tensor::attn);
    const auto b_acc = TensorAccessor(tensor::wo);
    const auto out = TensorAccessor(tensor::out);

    // Dst mode reloads the running total into DST before every k-block and packs it back
    // after, which costs O(output block) per k-block -- and that block is mt*nt tiles, 128
    // at mt=8 by nt=16. L1 mode lets the PACKER add into the partial
    // instead, so the total never enters DST at all: one pack per k-block rather than a
    // copy-in and a pack. See the numbers in unified_llama_prefill.md.
#if defined(MMB_ACC_DST)
    u::Accumulator<Out, u::AccumulatorMode::Dst> acc(acc_storage, out_storage);
#else
    u::Accumulator<Out, u::AccumulatorMode::L1> acc(acc_storage, out_storage);
#endif

#if defined(MMB_MCAST)
    // MULTICAST. Without it every core reads its own operands from DRAM, so the traffic term
    // scales with the core count and the whole thing stops improving: measured on
    // [512,2048]@[2048,2048], 16 cores reached 814us and 64 cores got WORSE, 1359us.
    //
    // The tiles a core needs are not its own, though. Cores in a grid ROW share an m-block,
    // so they all want the same A tiles; cores in a COLUMN share an n-block and all want the
    // same B tiles. So each tile is read from DRAM once and broadcast to the cores that
    // share it, and the traffic stops depending on how many cores there are.
    //
    // The price is a strict mapping: core (r, c) owns output block (r, c), so the grid must
    // be exactly mb x nb. A multicast is COLLECTIVE -- every core in the group has to make
    // the same calls in the same order, or the handshakes desynchronise -- which the flat
    // unit range below cannot promise, since split_evenly hands different cores different
    // counts.
    static_assert(
        MMB_GRID_H * MMB_GRID_W > 0 && (ntot / nt) == MMB_GRID_W,
        "with MMB_MCAST the grid width must equal the number of output-column blocks: core "
        "(r, c) owns block (r, c), so there is nowhere to put a spare block");

    const u::LogicalCoord me = u::LogicalCoord::this_core();
    // Every core in a row runs the row statement and every core in a column runs the column
    // statement; which side of each handshake it takes is decided inside, on its coordinate.
    const u::LogicalMcast row{u::LogicalCoord::yx(me.y, 0), u::Extent::hw(1, MMB_GRID_W)};
    const u::LogicalMcast col{u::LogicalCoord::yx(0, me.x), u::Extent::hw(MMB_GRID_H, 1)};

    {
        const uint32_t i = me.y;
        const uint32_t n = me.x;
        acc.clear();

#if defined(MMB_ABL_HOIST)
        // TIMING ABLATION ONLY -- deliberately the WRONG ANSWER. Every k-block reuses block
        // 0's operands, so the k-loop keeps its matmuls and its accumulator but loses every
        // per-k-block DRAM read and broadcast. The gap against the real kernel is what the
        // operand movement costs, which is the question once fidelity has shown the math is
        // not on the critical path at all.
        u::ComputeBlock a_h =
            u::noc_load<MMB_IN0_THREAD, 0>(a_storage, row, [&](u::L1Pages pages) {
                for (uint32_t p = 0; p < pages.count; ++p) {
                    const uint32_t rr = i * mt + p / kt;
                    const uint32_t cc = p % kt;
                    noc_async_read(a_acc.get_noc_addr(rr * ktot + cc), pages.addr(p), pages.page_bytes);
                }
            }).wait();
        u::ComputeBlock w_h =
            u::noc_load<MMB_IN1_THREAD, 1>(w_storage, col, [&](u::L1Pages pages) {
                for (uint32_t p = 0; p < pages.count; ++p) {
                    const uint32_t rr = p / nt;
                    const uint32_t cc = n * nt + p % nt;
                    noc_async_read(b_acc.get_noc_addr(rr * ntot + cc), pages.addr(p), pages.page_bytes);
                }
            }).wait();
#endif

        for (uint32_t b = 0; b < kb; ++b) {
            const bool finish = (b == kb - 1);

            // The two broadcasts take SEPARATE handshake pairs, and get them for free by
            // running on different threads: a core is the sender of one and a receiver of
            // the other, and sharing a ready counter across both would let one core's count
            // land while another is still waiting, so a wait-for-equality would never match.
            // Different threads also means different NOCs, so the two overlap.
#if defined(MMB_ABL_HOIST)
            const u::ComputeBlock<A>& a = a_h;
            const u::ComputeBlock<W>& w = w_h;
#else
            u::ComputeBlock a =
                u::noc_load<MMB_IN0_THREAD, /*pair=*/0>(a_storage, row, [&](u::L1Pages pages) {
                    for (uint32_t p = 0; p < pages.count; ++p) {
                        const uint32_t rr = i * mt + p / kt;
                        const uint32_t cc = b * kt + p % kt;
                        noc_async_read(a_acc.get_noc_addr(rr * ktot + cc), pages.addr(p), pages.page_bytes);
                    }
                }).wait();
            u::ComputeBlock w =
#if defined(MMB_SHARE_PAIR)
                u::noc_load<MMB_IN1_THREAD, /*pair=*/0>(w_storage, col, [&](u::L1Pages pages) {
#else
                u::noc_load<MMB_IN1_THREAD, /*pair=*/1>(w_storage, col, [&](u::L1Pages pages) {
#endif
                    for (uint32_t p = 0; p < pages.count; ++p) {
                        const uint32_t rr = b * kt + p / nt;
                        const uint32_t cc = n * nt + p % nt;
                        noc_async_read(b_acc.get_noc_addr(rr * ntot + cc), pages.addr(p), pages.page_bytes);
                    }
                }).wait();
#endif

#if defined(MMB_SKEW)
            // AFTER both loads, so `a` and `w` are still live -- their pops are at the end
            // of this iteration. Row 0's receivers therefore hold buffers they have not
            // freed while the next round's ready counting is under way, which is the only
            // placement that can turn an early broadcast into corruption. Before the loads
            // the delay sits after the previous pop, so the buffer is already free.
            //
            // Row 0's sender is (0, 0): delaying (0, 1..) parks it, while lower rows race
            // on into the column collective and increment (0, 0). volatile to survive -O3.
            if (me.y == 0 && me.x != 0) {
                for (volatile uint32_t d = 0; d < MMB_SKEW; ++d) {
                }
            }
#endif

            auto store_block = [&](u::Block<Out> blk) {
                u::noc_store<0>(std::move(blk), [&](u::L1Pages pages) {
                    for (uint32_t p = 0; p < pages.count; ++p) {
                        const uint32_t rr = i * mt + p / nt;
                        const uint32_t cc = n * nt + p % nt;
                        noc_async_write(pages.addr(p), out.get_noc_addr(rr * ntot + cc), pages.page_bytes);
                    }
                });
            };

            if constexpr (kb == 1) {
                (void)finish;
                store_block(out_storage.store(u::matmul(a, w)));
            } else {
                u::Block result = acc.accumulate(u::matmul(a, w), finish);
                if (finish) {
                    store_block(std::move(result));
                }
            }
        }
    }
#else

    for (uint32_t u = 0; u < block_count; ++u) {
        // Flat (m, n) index. m-major, so a core given consecutive units walks one row band's
        // column blocks before moving down.
        const uint32_t i = (block_begin + u) / nb;
        const uint32_t n = (block_begin + u) % nb;
        {
            acc.clear();

            // This block of the output: rows [i*mt, +mt) by columns [n*nt, +nt).
            auto store_block = [&](u::Block<Out> blk) {
                u::noc_store<0>(std::move(blk), [&](u::L1Pages pages) {
                    for (uint32_t p = 0; p < pages.count; ++p) {
                        const uint32_t row = i * mt + p / nt;
                        const uint32_t col = n * nt + p % nt;
                        noc_async_write(pages.addr(p), out.get_noc_addr(row * ntot + col), pages.page_bytes);
                    }
                });
            };

            for (uint32_t b = 0; b < kb; ++b) {
                const bool finish = (b == kb - 1);

                // A's (m, k) tile: rows [i*mt, +mt) by columns [b*kt, +kt). Strided, because A's
                // rows are what is contiguous.
                u::ComputeBlock a =
                    u::noc_load<MMB_IN0_THREAD>(a_storage, [&](u::L1Pages pages) {
                        for (uint32_t p = 0; p < pages.count; ++p) {
                            // Row-major in L1: page p is tile (p / kt, p % kt).
                            const uint32_t row = i * mt + p / kt;
                            const uint32_t col = b * kt + p % kt;
                            noc_async_read(a_acc.get_noc_addr(row * ktot + col), pages.addr(p), pages.page_bytes);
                        }
                    }).wait();

                // B's (k, n) tile: rows [b*kt, +kt) by columns [n*nt, +nt).
                u::ComputeBlock w =
                    u::noc_load<MMB_IN1_THREAD>(w_storage, [&](u::L1Pages pages) {
                        for (uint32_t p = 0; p < pages.count; ++p) {
                            const uint32_t row = b * kt + p / nt;
                            const uint32_t col = n * nt + p % nt;
                            noc_async_read(b_acc.get_noc_addr(row * ntot + col), pages.addr(p), pages.page_bytes);
                        }
                    }).wait();

                if constexpr (kb == 1) {
                    // One k-block: nothing to carry, so skip the accumulator entirely.
                    (void)finish;
                    store_block(out_storage.store(u::matmul(a, w)));
                } else {
                    u::Block result = acc.accumulate(u::matmul(a, w), finish);
                    if (finish) {
                        store_block(std::move(result));
                    }
                }
            }
        }
    }
#endif
}
