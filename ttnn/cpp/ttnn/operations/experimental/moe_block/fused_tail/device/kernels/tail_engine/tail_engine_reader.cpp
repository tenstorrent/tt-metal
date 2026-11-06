// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailEngine engine core, RISCV_0 on NoC0: every DRAM read (my W' K range, my own rows' shares, my se tiles) and the W'
// multicast down my half column.

#include "tail_engine_common.hpp"
#include "api/debug/kernel_profiler.hpp"

constexpr auto z_args = TensorAccessorArgs<11>();  // (the combine's output: this chip's owned rows, row-major)
constexpr auto s_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();

void kernel_main() {
    size_t a = 0;
    const uint32_t p = get_arg_val<uint32_t>(a++), pn = get_arg_val<uint32_t>(a++), kh0 = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++), wbank = get_arg_val<uint32_t>(a++),
                   woff = get_arg_val<uint32_t>(a++);
    const uint32_t wk0 = get_arg_val<uint32_t>(a++), wk1 = get_arg_val<uint32_t>(a++);
    const uint32_t mx = get_arg_val<uint32_t>(a++), yt = get_arg_val<uint32_t>(a++), yb = get_arg_val<uint32_t>(a++);
    const uint32_t z_addr = get_arg_val<uint32_t>(a++), ak0 = get_arg_val<uint32_t>(a++),
                   ak1 = get_arg_val<uint32_t>(a++);
    const uint32_t se_addr = get_arg_val<uint32_t>(a++), NTO = get_arg_val<uint32_t>(a++),
                   f0 = get_arg_val<uint32_t>(a++);
    const uint32_t fn = get_arg_val<uint32_t>(a++), nst = get_arg_val<uint32_t>(a++), nown = get_arg_val<uint32_t>(a++);
    // (the W' loads' gate: the lead core (its NoC), the engine's NoC rectangle, its cores, am I the lead)
    const uint32_t gw = get_arg_val<uint32_t>(a++), gx0 = get_arg_val<uint32_t>(a++), gy0 = get_arg_val<uint32_t>(a++);
    const uint32_t gx1 = get_arg_val<uint32_t>(a++), gy1 = get_arg_val<uint32_t>(a++), gn = get_arg_val<uint32_t>(a++);
    const uint32_t glead = get_arg_val<uint32_t>(a++);
    const uint32_t npz = get_arg_val<uint32_t>(a++), pz = a;  // (the lead: the port cores it releases, x | y << 16)
    const uint32_t ord = pz + npz;
    {  // this call's step words zeroed here, then counted in on the lead; the lead releases the writers and the ports
        DeviceZoneScopedN("TL-ZGATE");
        volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WD));
        for (uint32_t i = 0; i < WD_BYTES / 4; ++i) {
            wd[i] = 0;
        }
        noc_semaphore_inc(get_noc_addr(gw & 0xFFFF, gw >> 16, get_semaphore(SZ)), 1);
        if (glead) {
            volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SZ));
            noc_semaphore_wait(cnt, gn);
            *cnt = 0;  // (re-arm: every core's one increment landed)
            // (the go's source: the words' spare slot (kind 3, step 0), not my SZGO - my writer re-arms that one,
            // possibly before the multicast has read it)
            wd[4 * 3 * NSTMAX] = 1;
            noc_semaphore_set_multicast_loopback_src(
                reinterpret_cast<uint32_t>(wd + 4 * 3 * NSTMAX),
                get_noc_multicast_addr(gx0, gy0, gx1, gy1, get_semaphore(SZGO)),
                gn);
            for (uint32_t i = 0; i < npz; ++i) {
                const uint32_t xy = get_arg_val<uint32_t>(pz + i);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, get_semaphore(PZR)), 1);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, get_semaphore(PZW)), 1);
            }
            noc_async_write_barrier();
            noc_async_atomic_barrier();
        }
    }
    const auto zg = TensorAccessor(z_args, z_addr, KT * 64);  // (a page: one latent row)
    const auto sg = TensorAccessor(s_args, se_addr, 2048);
    cb_reserve_back(CB_ONES, 1);  // (the all-ones tile of the squares' row sums)
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ONES));
        for (uint32_t i = 0; i < 512; ++i) {
            o[i] = 0x3F803F80u;
        }
    }
    cb_push_back(CB_ONES, 1);
    const uint32_t n = ak1 - ak0;
    // my share of my own rows, row-major (the combine's output: the 32 rows of row tile t, K tiles kh0 + ak0 ..), own
    // step by own step: the compute tilizes each into its landing slot (the other steps' shares land by the fabric)
    auto own = [&](uint32_t j) {
        const uint32_t t = step_t(j, p);
        cb_reserve_back(CB_RM, n);
        const uint32_t dst = get_write_ptr(CB_RM);
        for (uint32_t i = 0; i < 32; ++i) {
            noc_async_read(zg.get_noc_addr(t * 32 + i, (kh0 + ak0) * 64), dst + i * n * 64, n * 64);
        }
        noc_async_read_barrier();
        cb_push_back(CB_RM, n);
    };
    if (nown > 0) {
        DeviceZoneScopedN("TL-RMREAD");
        own(0);  // (the NCRISC multicasts it while W' loads)
    }
    {  // every engine core's first own rows read before any W' load: the rows' small strided reads queued behind the
       // W' bursts in the DRAM took 13-26 us, holding the gather's first sends (the ports read the tilized rows)
        DeviceZoneScopedN("TL-WGATE");
        noc_semaphore_inc(get_noc_addr(gw & 0xFFFF, gw >> 16, get_semaphore(SG)), 1);
        volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SGO));
        if (glead) {
            volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SG));
            noc_semaphore_wait(cnt, gn);
            *cnt = 0;  // (re-arm: every core's one increment landed)
            *go = 1;
            noc_semaphore_set_multicast_loopback_src(
                get_semaphore(SGO), get_noc_multicast_addr(gx0, gy0, gx1, gy1, get_semaphore(SGO)), gn);
            noc_async_write_barrier();
        }
        while (*go == 0) {
            invalidate_l1_cache();
        }
        *go = 0;  // (re-arm: one set a call)
    }
    {  // W': my K range of my half column's [K half][pn] (one contiguous bank region: one burst), then down the half
       // column
        DeviceZoneScopedN("TL-WLOAD");
        // (in K order a wave at a time: my half column's range p - 1 in first - the chip's DRAM serves the ranges the
        // compute takes next, not all 110 bursts at once)
        if (p > 0) {
            volatile tt_l1_ptr uint32_t* swv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SW));
            while (!((*swv >> (p - 1)) & 1)) {
                invalidate_l1_cache();
            }
        }
        const uint32_t src = get_write_ptr(CB_W) + wk0 * pn * WTB, bytes = (wk1 - wk0) * pn * WTB;
        noc_async_read(get_noc_addr_from_bank_id<true>(wbank, w_addr + woff), src, bytes);
        noc_async_read_barrier();
        if (PAIRS > 1) {
            if (bytes) {
                noc_async_write_multicast(src, get_noc_multicast_addr(mx, yt, mx, yb, src), bytes, PAIRS - 1);
            }
            noc_async_writes_flushed();
            noc_semaphore_inc_multicast(get_noc_multicast_addr(mx, yt, mx, yb, get_semaphore(SW)), 1u << p, PAIRS - 1);
        }
        noc_semaphore_inc(get_noc_addr(get_semaphore(SW)), 1u << p);
    }
    for (uint32_t j = 1; j < nown; ++j) {
        own(j);
    }
    for (uint32_t s = 0; s < nst; ++s) {  // my se tiles (the columns I finish), step by step
        const uint32_t t = step_t(s, p), R = get_arg_val<uint32_t>(ord + t / RT) * RT + t % RT;
        cb_reserve_back(CB_SE, fn);
        const uint32_t dst = get_write_ptr(CB_SE);
        for (uint32_t j = 0; j < fn; ++j) {
            noc_async_read(sg.get_noc_addr(R * NTO + f0 + j), dst + j * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(CB_SE, fn);
    }
    noc_async_atomic_barrier();
}
