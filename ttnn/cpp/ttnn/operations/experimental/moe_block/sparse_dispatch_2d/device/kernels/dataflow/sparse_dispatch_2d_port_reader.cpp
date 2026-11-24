// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_dispatch_2d_port_reader: feeds a dispatch port's sender the rows to forward one hop over its link
// Runs on: dispatch port cores (one per direction with a neighbour and per link), RISCV_1 (NCRISC), NoC 0
// Responsibilities:
//  - reads the route plan and each row to send from xbuf (`out`, DRAM) over NoC 0; polls the manifest rings (L1)
//  - merges own rows (trees starting in its direction) with the input rings (continue, turn +, turn -) and END(g)s
//  - pushes rows to CB_ROWS and 16-byte entries to CB_META `group` at a time for sparse_dispatch_2d_port_writer
//  - waits on the ring counters (bumped over the fabric by the neighbours' senders), resets them at the end
// Note: rows land in xbuf through the fabric router's NoC-1 DRAM endpoint and are read here through NoC 0's. The BH
//       DRAM cross-port stale read needs small pages under heavy DRAM background reads; these reads are whole rows.
//
// The 16-byte metadata page of each row has word 0 as the manifest entry and word 1 the mask.
//
// The main loop makes one step a pass, the first that applies, in this order:
//  1. Drain the rings: take at most one entry from each input ring whose counter is ahead of what was consumed. A row
//     entry is read from xbuf into the batch; an END entry only records its generation in end_in[k].
//  2. Only if no ring had an entry: start the next own row of this port (own_t walks the S own rows once; a row that
//     does not use this port's direction and link is skipped without a step).
//  3. Only if there was neither and every own row is looked at: emit END(g) for g = end_out + 1 once every ordering
//     input has delivered END(g - 1) (END(0) has no condition). After END(GMAX) is out, wait for every input's
//     END(GMAX) and leave the loop.
//  4. Only if nothing above applied: flush a partial batch, so the sender is never starved of rows that are already
//     read while this port waits for the neighbours.
// After the loop: flush, and give the counters back by what was consumed.
//
// Why it is built this way:
//  - Rings come before own rows: forwarded rows are what the downstream chips' trees wait on. Own rows only wait
//    while ring entries are pending, and every ring carries a finite number of entries (it ends with END(GMAX)), so
//    own rows are delayed, never starved. No fairer mix is implemented.
//  - A ring entry may be read as soon as its counter is ahead. The neighbour's sender puts the entries after the rows
//    they describe on the same fabric connection, and raises the counter with the fused write + atomic increment that
//    carries the entries; a connection delivers in order, so the counter rising means the entries are in the ring and
//    the rows sent before them have landed in xbuf. That is why a row is read from xbuf right after its entry,
//    without another wait.
//  - END carries a generation. The ports form cycles on a torus (a port's output feeds the neighbours whose outputs
//    feed it), so waiting for an END from every input before sending one would deadlock. END(g) waits only for
//    END(g - 1) of the inputs: END(0) waits for nothing, which breaks the cycle. An entry's generation is the number of
//    hops its row has taken, a row takes at most (C - 1) + (R - 1) hops and the last hop makes no entry, so the last
//    generation is GMAX = (C - 1) + (R - 1) - 1.
//  - END(GMAX) is the last entry every sender writes on a ring, and the reader leaves only after END(GMAX) of every
//    ring, so its counters then hold exactly what it consumed. Subtracting that (an atomic add of -consumed, not a
//    write of 0) leaves them at zero for the next call, as the persistent words must be.
//
// State: ring_consumed[k] entries taken from ring k (the next entry's index); end_in[k] the last END generation
// seen on ring k (-1: none); end_out the last END generation emitted (-1: none); own_t the next own row to try;
// `did` whether the pass made a step.
//
// Compile-time args: named (sparse_dispatch_2d_names.hpp): manifest_cb (this port core's input rings, at the same
// address on every port core of every chip: the program's first CB), rows_cb, meta_cb, plan_cb, row_bytes, group,
// plan_bytes and the mesh constants; then the TensorAccessorArgs of out, plan at offset 0.
// Runtime args (sparse_dispatch_2d_layout.hpp reader_arg): out, plan, words (the port's semaphore words), ring_offs,
// ring_mask, my_r, my_c, dir, link: the core's block in the DRAM args table (common/args_table.hpp).

#include "sparse_dispatch_2d_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"

namespace sparse_dispatch {

struct ReaderArgs {
    uint32_t out_addr;
    uint32_t plan_addr;
    uint32_t man_addr;  // this port core's manifest rings (the same address on every port core of every chip)
    uint32_t ring_offs;
    uint32_t cnt_addr[3];
    // which inputs exist (bit k: ring k); bit 3 + k: ring k is a terminal stream (only the link's END(GMAX): it gates
    // this port's completion - the link's rows landed - not its END order)
    uint32_t ring_mask;
    uint32_t my_r;
    uint32_t my_c;
    uint32_t dir;  // 0 X+, 1 X-, 2 Y+, 3 Y-
    uint32_t link;
};

FORCE_INLINE ReaderArgs reader_args(uint32_t man_addr) {
    namespace reader_arg = lay::reader_arg;
    ReaderArgs r{};
    r.out_addr = moe_block_args::arg(reader_arg::kOut);
    r.plan_addr = moe_block_args::arg(reader_arg::kPlan);
    const uint32_t words = moe_block_args::arg(reader_arg::kWords);
    r.man_addr = man_addr;
    r.ring_offs = moe_block_args::arg(reader_arg::kRingOffsets);
    for (uint32_t k = 0; k < 3; ++k) {
        r.cnt_addr[k] = words + k * SEM_STRIDE;
    }
    r.ring_mask = moe_block_args::arg(reader_arg::kRingMask);
    r.my_r = moe_block_args::arg(reader_arg::kMyRow);
    r.my_c = moe_block_args::arg(reader_arg::kMyCol);
    r.dir = moe_block_args::arg(reader_arg::kDir);
    r.link = moe_block_args::arg(reader_arg::kLink);
    return r;
}

// ring k's counter, as the neighbour's sender left it: a plain L1 read of a persistent word, so the caller invalidates
// the L1 cache first (once a pass: it polls the three counters without blocking, which no semaphore wait can do)
FORCE_INLINE uint32_t entries_arrived(volatile tt_l1_ptr uint32_t* cnt) { return *cnt; }

// whether every input in `mask` (bit k: ring k) has delivered END(g) or later
FORCE_INLINE bool inputs_reached(const int32_t (&end_in)[3], int32_t g, uint32_t mask) {
    for (uint32_t k = 0; k < 3; ++k) {
        if (((mask >> k) & 1) && end_in[k] < g) {
            return false;
        }
    }
    return true;
}

// The hops this port gives the own rows of the chip: read-only, from the route plan.
struct OwnRows {
    const uint32_t* words;  // the route plan
    uint32_t my_r, my_c;
    bool axis_x, plus;
    uint32_t link;

    // hops this port gives own row t (0: not this port's); sets the destination mask `m` (own chip removed) and the
    // dimension order
    FORCE_INLINE uint32_t hops(uint32_t t, uint32_t& m, uint32_t& order) const {
        const uint32_t f = route_flags(words, t);
        if (lay::kFlagLink.get(f) != link) {
            return 0;
        }
        m = route_mask(words, t) & ~(1u << (my_r * C + my_c));
        if (m == 0) {
            return 0;
        }
        order = lay::kFlagOrder.get(f);  // 0: X first
        const bool first_x = order == 0;
        // my axis is the first leg, or the turn at the origin into my axis
        const uint32_t span =
            axis_x == first_x ? first_leg_span(axis_x, my_r, my_c, m, t) : turn_span(axis_x, my_r, my_c, m, t);
        return plus ? lay::kSpanForward.get(span) : lay::kSpanBackward.get(span);
    }
};

// The rows and metadata pages being filled for the sender: `group` of each, pushed to the sender together. A group's
// pages are reserved when its first row is put and pushed (after the row reads land) when it is full or flushed.
// put / put_end / flush are the only places that reserve, push or read from xbuf.
template <uint32_t cb_rows, uint32_t cb_meta, uint32_t group, uint32_t row_bytes>
struct RowBatch {
    uint32_t filled = 0;  // pages of the open group already written
    uint32_t rows_w = 0;  // the open group's first row page / metadata page (L1)
    uint32_t meta_w = 0;

    // pad the open group's unused metadata pages with SKIP, wait for its row reads, and push it to the sender
    FORCE_INLINE void flush() {
        if (filled) {
            for (uint32_t i = filled; i < group; ++i) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w)[i * META_WORDS] = SKIP;
            }
            noc_async_read_barrier();
            cb_push_back(cb_rows, group);
            cb_push_back(cb_meta, group);
            filled = 0;
        }
    }
    // may block on the sender freeing a group: reserves both CBs' pages of a new group
    FORCE_INLINE void open_group() {
        if (filled == 0) {
            cb_reserve_back(cb_rows, group);
            rows_w = get_write_ptr(cb_rows);
            cb_reserve_back(cb_meta, group);
            meta_w = get_write_ptr(cb_meta);
        }
    }
    // read the row at `slot` of xbuf into the next page, with the entry words (may reserve, may push a full group)
    template <typename OUT>
    FORCE_INLINE void put(const OUT& out, uint32_t slot, uint32_t w0, uint32_t w1) {
        open_group();
        noc_async_read(out.get_noc_addr(slot), rows_w + filled * row_bytes, row_bytes);
        volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w) + filled * META_WORDS;
        mp[0] = w0;
        mp[1] = w1;
        if (++filled == group) {
            flush();
        }
    }
    // an END(g) entry (no row; its row page is not written)
    FORCE_INLINE void put_end(uint32_t g) {
        open_group();
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w)[filled * META_WORDS] = lay::end_word0(g);
        if (++filled == group) {
            flush();
        }
    }
};

void read(const ReaderArgs& args) {
    constexpr uint32_t cb_rows = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kRowsCb);
    constexpr uint32_t cb_meta = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kMetaCb);
    constexpr uint32_t cb_plan = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kPlanCb);
    constexpr uint32_t row_bytes = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kRowBytes);
    constexpr uint32_t group = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kGroup);
    constexpr uint32_t PLAN_BYTES = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kPlanBytes);
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto plan_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t ring_offs = args.ring_offs, ring_mask = args.ring_mask;
    const uint32_t my_r = args.my_r, my_c = args.my_c, dir = args.dir, link = args.link;
    const uint32_t me = my_r * C + my_c;
    const bool axis_x = dir < 2;
    const bool plus = (dir & 1) == 0;
    const auto out = TensorAccessor(out_args, args.out_addr, row_bytes);
    const auto plan = TensorAccessor(plan_args, args.plan_addr, PLAN_BYTES);

    cb_reserve_back(cb_plan, 1);
    const uint32_t plan_l1 = get_write_ptr(cb_plan);
    noc_async_read(plan.get_noc_addr(0), plan_l1, PLAN_BYTES);
    noc_async_read_barrier();
    const uint32_t* words = reinterpret_cast<const uint32_t*>(plan_l1);

    volatile tt_l1_ptr uint32_t* cnt[3];
    volatile tt_l1_ptr uint32_t* ring[3];
    for (uint32_t k = 0; k < 3; ++k) {
        cnt[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.cnt_addr[k]);
        ring[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            args.man_addr + lay::ring_offset(ring_offs, k) * lay::kEntryBytes);
    }
    uint32_t ring_consumed[3] = {0, 0, 0};
    int32_t end_in[3] = {-1, -1, -1};
    int32_t end_out = -1;
    uint32_t own_t = 0;

    const OwnRows own{words, my_r, my_c, axis_x, plus, link};
    RowBatch<cb_rows, cb_meta, group, row_bytes> batch;
    const uint32_t order_mask = ring_mask & ~(ring_mask >> 3) & 7;  // the inputs END(g) waits on

    while (true) {
        bool did = false;
        invalidate_l1_cache();
        for (uint32_t k = 0; k < 3; ++k) {
            if (((ring_mask >> k) & 1) && ring_consumed[k] < entries_arrived(cnt[k])) {
                volatile tt_l1_ptr uint32_t* entry = &ring[k][lay::kEntryWords * ring_consumed[k]];
                const uint32_t word0 = entry[0], word1 = entry[1];
                ++ring_consumed[k];
                if (word0 & END) {
                    end_in[k] = static_cast<int32_t>(lay::kGeneration.get(word0));
                } else {
                    batch.put(out, lay::kSlot.get(word0), word0, word1);
                }
                did = true;
            }
        }
        if (!did) {
            while (own_t < S) {
                const uint32_t t = own_t++;
                uint32_t m = 0, order = 0;
                const uint32_t h = own.hops(t, m, order);
                if (h) {
                    const uint32_t slot = me * S + t;
                    batch.put(out, slot, lay::entry_word0(slot, h, 0, order), m);
                    did = true;
                    break;
                }
            }
        }
        if (!did && own_t >= S) {
            const int32_t g = end_out + 1;
            if (g <= static_cast<int32_t>(GMAX) && (g == 0 || inputs_reached(end_in, g - 1, order_mask))) {
                batch.put_end(g);
                end_out = g;
                did = true;
            } else if (g > static_cast<int32_t>(GMAX) && inputs_reached(end_in, GMAX, ring_mask)) {
                break;
            }
        }
        if (!did) {
            batch.flush();
        }
    }
    batch.flush();
    // the counters were raised by the neighbours' senders (END(GMAX) was their last entry): give back what was taken
    for (uint32_t k = 0; k < 3; ++k) {
        if (ring_consumed[k]) {
            noc_semaphore_inc(get_noc_addr(args.cnt_addr[k]), 0u - ring_consumed[k]);
        }
    }
    noc_async_atomic_barrier();
}

}  // namespace sparse_dispatch

void kernel_main() {
    constexpr uint32_t cb_man = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kManifestCb);
    moe_block_args::load();  // (this core's values: from the DRAM table, common/args_table.hpp)
    // (a port core of another chip only: one program for the mesh)
    if (moe_block_args::arg(sparse_dispatch::lay::reader_arg::kDir) == sparse_dispatch::IDLE_PORT) {
        return;
    }
    sparse_dispatch::read(sparse_dispatch::reader_args(get_write_ptr(cb_man)));
}
