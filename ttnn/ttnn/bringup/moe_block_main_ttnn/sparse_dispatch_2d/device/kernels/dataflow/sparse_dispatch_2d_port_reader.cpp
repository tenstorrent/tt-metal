// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The dispatch's port reader (RISCV_1, NoC0): merge the input rings (entries whose rows have landed) and the own rows
// this port starts, hand them to the sender `group` at a time with a 16-byte metadata page each (word 0 as the
// manifest entry, word 1 the mask).
//
// Compile-time args: 0 the manifest CB (this port core's input rings, at the same address on every port core of every
// chip: the program's first CB), 1 cb_rows, 2 cb_meta, 3 cb_plan, 4 row_bytes, 5 group, 6 plan_bytes, then the
// TensorAccessorArgs of out, plan.
// Runtime args: out, plan, words (the port's semaphore words), ring_offs, ring_mask, my_r, my_c, dir, lnk.

#include "sparse_dispatch_2d_common.hpp"

namespace sa {

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
    uint32_t lnk;
};

FORCE_INLINE ReaderArgs reader_args(size_t& a, uint32_t man_addr) {
    ReaderArgs r{};
    r.out_addr = get_arg_val<uint32_t>(a++);
    r.plan_addr = get_arg_val<uint32_t>(a++);
    const uint32_t words = get_arg_val<uint32_t>(a++);
    r.man_addr = man_addr;
    r.ring_offs = get_arg_val<uint32_t>(a++);
    for (uint32_t k = 0; k < 3; ++k) {
        r.cnt_addr[k] = words + k * SEM_STRIDE;
    }
    r.ring_mask = get_arg_val<uint32_t>(a++);
    r.my_r = get_arg_val<uint32_t>(a++);
    r.my_c = get_arg_val<uint32_t>(a++);
    r.dir = get_arg_val<uint32_t>(a++);
    r.lnk = get_arg_val<uint32_t>(a++);
    return r;
}

void read(const ReaderArgs& args) {
    constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t cb_plan = get_compile_time_arg_val(3);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);
    constexpr uint32_t PLAN_BYTES = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr auto plan_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t ring_offs = args.ring_offs, ring_mask = args.ring_mask;
    const uint32_t my_r = args.my_r, my_c = args.my_c, dir = args.dir, lnk = args.lnk;
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
        ring[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.man_addr + ring_off(ring_offs, k) * 16);
    }
    uint32_t mi[3] = {0, 0, 0};
    int32_t end_in[3] = {-1, -1, -1};
    int32_t end_out = -1;
    uint32_t own_t = 0, batch = 0, rows_w = 0, meta_w = 0;

    // hops this port gives own row t (0: not this port's)
    auto own_hops = [&](uint32_t t, uint32_t& m, uint32_t& order) -> uint32_t {
        const uint32_t f = route_flags(words, t);
        if (((f >> 1) & 1) != lnk) {
            return 0;
        }
        m = route_mask(words, t) & ~(1u << me);
        if (m == 0) {
            return 0;
        }
        order = f & 1;  // 0: X first
        const bool first_x = order == 0;
        uint32_t ab;
        if (axis_x == first_x) {  // my axis is the first leg
            ab = axis_x ? xspan(my_c, cols_of(m), t) : yspan(my_r, rows_of(m), t);
        } else {  // the turn at the origin into my axis
            ab = axis_x ? xspan(my_c, cols_in_row(m, my_r), t) : yspan(my_r, rows_in_col(m, my_c), t);
        }
        return plus ? (ab & 15) : (ab >> 4);
    };
    auto flush = [&]() {
        if (batch) {
            for (uint32_t i = batch; i < group; ++i) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w)[i * META_WORDS] = SKIP;
            }
            noc_async_read_barrier();
            cb_push_back(cb_rows, group);
            cb_push_back(cb_meta, group);
            batch = 0;
        }
    };
    auto slot_page = [&]() {
        if (batch == 0) {
            cb_reserve_back(cb_rows, group);
            rows_w = get_write_ptr(cb_rows);
            cb_reserve_back(cb_meta, group);
            meta_w = get_write_ptr(cb_meta);
        }
    };
    auto put = [&](uint32_t slot, uint32_t w0, uint32_t w1) {
        slot_page();
        noc_async_read(out.get_noc_addr(slot), rows_w + batch * row_bytes, row_bytes);
        volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w) + batch * META_WORDS;
        mp[0] = w0;
        mp[1] = w1;
        if (++batch == group) {
            flush();
        }
    };
    auto put_end = [&](uint32_t g) {
        slot_page();
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w)[batch * META_WORDS] = END | (g << 20);
        if (++batch == group) {
            flush();
        }
    };
    auto inputs_reached = [&](int32_t g, uint32_t mask) {
        for (uint32_t k = 0; k < 3; ++k) {
            if (((mask >> k) & 1) && end_in[k] < g) {
                return false;
            }
        }
        return true;
    };
    const uint32_t order_mask = ring_mask & ~(ring_mask >> 3) & 7;  // the inputs END(g) waits on

    while (true) {
        bool did = false;
        invalidate_l1_cache();
        for (uint32_t k = 0; k < 3; ++k) {
            if (((ring_mask >> k) & 1) && mi[k] < *cnt[k]) {
                const uint32_t e0 = ring[k][4 * mi[k]], e1 = ring[k][4 * mi[k] + 1];
                ++mi[k];
                if (e0 & END) {
                    end_in[k] = static_cast<int32_t>((e0 >> 20) & 0xF);
                } else {
                    put(e0 & 0xFFFF, e0, e1);
                }
                did = true;
            }
        }
        if (!did) {
            while (own_t < S) {
                const uint32_t t = own_t++;
                uint32_t m = 0, order = 0;
                const uint32_t h = own_hops(t, m, order);
                if (h) {
                    const uint32_t slot = me * S + t;
                    put(slot, slot | (h << 16) | (order << 24), m);
                    did = true;
                    break;
                }
            }
        }
        if (!did && own_t >= S) {
            const int32_t g = end_out + 1;
            if (g <= static_cast<int32_t>(GMAX) && (g == 0 || inputs_reached(g - 1, order_mask))) {
                put_end(g);
                end_out = g;
                did = true;
            } else if (g > static_cast<int32_t>(GMAX) && inputs_reached(GMAX, ring_mask)) {
                break;
            }
        }
        if (!did && batch) {
            flush();
        }
    }
    flush();
    for (uint32_t k = 0; k < 3; ++k) {
        if (mi[k]) {
            noc_semaphore_inc(get_noc_addr(args.cnt_addr[k]), 0u - mi[k]);
        }
    }
    noc_async_atomic_barrier();
}

}  // namespace sa

void kernel_main() {
    constexpr uint32_t cb_man = get_compile_time_arg_val(0);
    size_t a = 0;
    sa::read(sa::reader_args(a, get_write_ptr(cb_man)));
}
