// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#ifdef BHL_PROF
#include "api/debug/kernel_profiler.hpp"
#define BHLZ(n) DeviceZoneScopedN(n)
#else
#define BHLZ(n)
#endif

#if defined(BHL_MCASTM) || defined(BHL_MCASTR) || defined(BHL_LRMSM)
template <typename Src>
inline void bhl_mcast(
    const Noc& noc, const Src& src, std::uint32_t bytes, std::uint32_t addr, std::uint32_t x0, std::uint32_t y0,
    std::uint32_t x1, std::uint32_t y1, std::uint32_t gx0, std::uint32_t gx1, std::uint32_t sx, std::uint32_t sy,
    Semaphore<>* sem) {
    std::uint32_t rx0[2], rx1[2], nr = 0;
    if (gx1 > gx0 && gx0 > x0 && gx0 <= x1) {
        rx0[0] = x0; rx1[0] = gx0 - 1; rx0[1] = gx1; rx1[1] = x1; nr = 2;
    } else {
        rx0[0] = x0; rx1[0] = x1; nr = 1;
    }
    for (std::uint32_t k = 0; k < nr; ++k) {
        std::uint32_t xs = rx0[k], xe = rx1[k], ys = y0, ye = y1;
        const std::uint32_t n = (xe - xs + 1) * (ye - ys + 1);
        const bool incl = sx >= xs && sx <= xe && sy >= ys && sy <= ye;
        if (noc.get_noc_id() == 1) {
            std::uint32_t t = xs; xs = xe; xe = t;
            t = ys; ys = ye; ye = t;
        }
        if (sem == nullptr) {
            if (incl) {
                noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                    src, MulticastEndpoint{}, bytes, n, {.offset_bytes = 0},
                    {.noc_x_start = xs, .noc_y_start = ys, .noc_x_end = xe, .noc_y_end = ye, .addr = addr});
            } else {
                noc.async_write_multicast(
                    src, MulticastEndpoint{}, bytes, n, {.offset_bytes = 0},
                    {.noc_x_start = xs, .noc_y_start = ys, .noc_x_end = xe, .noc_y_end = ye, .addr = addr});
            }
        } else {
            if (incl) {
                sem->template set_multicast<NocOptions::MCAST_INCL_SRC>(noc, xs, ys, xe, ye, n);
            } else {
                sem->set_multicast(noc, xs, ys, xe, ye, n);
            }
        }
    }
}
#endif

void kernel_main() {
    const std::uint32_t c = get_arg(args::c);
    const std::uint32_t active = get_arg(args::active);
    const bool is_comb = get_arg(args::is_comb) == 1;
    const std::uint32_t comb_x = get_arg(args::comb_x);
    const std::uint32_t comb_y = get_arg(args::comb_y);
    const std::uint32_t x0 = get_arg(args::x0);
    const std::uint32_t y0 = get_arg(args::y0);
    const std::uint32_t x1 = get_arg(args::x1);
    const std::uint32_t y1 = get_arg(args::y1);
    const std::uint32_t rect_n = get_arg(args::rect_n);
    const std::uint32_t ck0 = get_arg(args::ck0);
    const std::uint32_t nck = get_arg(args::nck);
    const std::uint32_t nchunk = get_arg(args::nchunk);
    const std::uint32_t T = get_arg(args::T);
    const std::uint32_t t0 = get_arg(args::t0);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const std::uint32_t blk = get_arg(args::blk);
    const std::uint32_t row0 = get_arg(args::row0);
    const std::uint32_t rstride = get_arg(args::rstride);
    const std::uint32_t gx0 = get_arg(args::gx0);
    const std::uint32_t gx1 = get_arg(args::gx1);
    [[maybe_unused]] const std::uint32_t cj = get_arg(args::cj);
    [[maybe_unused]] const std::uint32_t cS = get_arg(args::cS);
    [[maybe_unused]] const std::uint32_t next_x = get_arg(args::next_x);
    [[maybe_unused]] const std::uint32_t next_y = get_arg(args::next_y);
    const std::uint32_t nsend = get_arg(args::nsend);
    [[maybe_unused]] const std::uint32_t snd = get_arg(args::snd);
    [[maybe_unused]] const std::uint32_t rcv = get_arg(args::rcv);
    [[maybe_unused]] const std::uint32_t sfirst = get_arg(args::sfirst);
    [[maybe_unused]] const std::uint32_t nps = get_arg(args::nps);
    [[maybe_unused]] const std::uint32_t pk0 = get_arg(args::pk0);
    [[maybe_unused]] const std::uint32_t pk1 = get_arg(args::pk1);
    [[maybe_unused]] const std::uint32_t lead_x = get_arg(args::lead_x);
    [[maybe_unused]] const std::uint32_t lead_y = get_arg(args::lead_y);
    [[maybe_unused]] const bool is_lead = get_arg(args::is_lead) == 1;
    [[maybe_unused]] const std::uint32_t gsz = get_arg(args::gsz);
    [[maybe_unused]] const std::uint32_t gslot = get_arg(args::gslot);
    [[maybe_unused]] const std::uint32_t gidx = get_arg(args::gidx);
    [[maybe_unused]] const std::uint32_t ngrp = get_arg(args::ngrp);
    if (T == 0) {
        return;
    }

    const Noc noc;
    DataflowBuffer out0(dfb::out0);
    DataflowBuffer pmax(dfb::pmax);
    DataflowBuffer rmax(dfb::rmax);
    DataflowBuffer mout(dfb::mout);
    DataflowBuffer mcb(dfb::mcb);
    DataflowBuffer psum(dfb::psum);
    DataflowBuffer rsum(dfb::rsum);
    DataflowBuffer rout(dfb::rout);
    DataflowBuffer rcb(dfb::rcb);
    [[maybe_unused]] DataflowBuffer pacc(dfb::pacc);
    [[maybe_unused]] DataflowBuffer racc(dfb::racc);
    [[maybe_unused]] DataflowBuffer rmax2(dfb::rmax2);
    [[maybe_unused]] Semaphore sem_rmax2(sem::rmax2_sem);
    [[maybe_unused]] Semaphore sem_acc(sem::acc_sem);
    Semaphore sem_rmax(sem::rmax_sem);
    Semaphore sem_rsum(sem::rsum_sem);
    Semaphore sem_m(sem::m_sem);
    Semaphore sem_r(sem::r_sem);
    const auto dst = TensorAccessor(tensor::dst);
    const std::uint32_t out_bytes = out0.get_entry_size();
    const std::uint32_t im_bytes = pmax.get_entry_size();
    const std::uint32_t acc_bytes = psum.get_entry_size();
#if defined(BHL_LRASTR) || defined(BHL_LRMSM)
    DataflowBuffer f5fl(dfb::f5fl);
    const std::uint32_t bhl_fb = f5fl.get_write_ptr();         // tile 0: chunk-sum flags, tile 1: max flags
    const std::uint32_t bhl_fsrc = bhl_fb + 2 * acc_bytes - 16;  // local word = 1, source of every flag write
    reinterpret_cast<volatile std::uint32_t*>(bhl_fsrc)[0] = 1u;
    auto bhl_flag = [&](std::uint32_t addr) {
        noc.async_write(
            CoreLocalMem<std::uint32_t>(bhl_fsrc), UnicastEndpoint{}, 16, {.offset_bytes = 0},
            {.noc_x = comb_x, .noc_y = comb_y, .addr = addr});
    };
    auto bhl_poll = [&](std::uint32_t addr) {
        volatile std::uint32_t* p = reinterpret_cast<volatile std::uint32_t*>(addr);
        while (p[0] == 0u) {
        }
    };
#endif
    // acc tile → (x, y, addr); with face_pair only faces 0 and 2 (column 0 lives there)
    auto bhl_put = [&](std::uint32_t src_addr, std::uint32_t x, std::uint32_t y, std::uint32_t addr, bool face_pair) {
        if (face_pair) {
            const std::uint32_t fb = acc_bytes / 4;
            noc.async_write(CoreLocalMem<std::uint32_t>(src_addr), UnicastEndpoint{}, fb, {.offset_bytes = 0}, {.noc_x = x, .noc_y = y, .addr = addr});
            noc.async_write(CoreLocalMem<std::uint32_t>(src_addr + 2 * fb), UnicastEndpoint{}, fb, {.offset_bytes = 0}, {.noc_x = x, .noc_y = y, .addr = addr + 2 * fb});
        } else {
            noc.async_write(CoreLocalMem<std::uint32_t>(src_addr), UnicastEndpoint{}, acc_bytes, {.offset_bytes = 0}, {.noc_x = x, .noc_y = y, .addr = addr});
        }
    };
#ifdef BHL_PCOL
    constexpr bool bhl_pcol = true;
#else
    constexpr bool bhl_pcol = false;
#endif
#ifdef BHL_CCOL
    constexpr bool bhl_ccol = true;
#else
    constexpr bool bhl_ccol = false;
#endif
#ifdef BHL_WDNOC
    const Noc noc_alt(static_cast<uint8_t>(1 - noc.get_noc_id()));
#endif
#ifdef BHL_LRACMP
    DataflowBuffer f5cp(dfb::f5cp);
    DataflowBuffer f5co(dfb::f5co);
#endif
    // flag word inside the chunk-sum tile: last word actually sent (column 0 never reaches it)
    [[maybe_unused]] const std::uint32_t bhl_fw = bhl_pcol ? (3 * acc_bytes / 4 - 4) : (acc_bytes - 4);
    // send one chunk sum to the combiner's slot (data + arrival signal in whichever form is compiled in)
    auto bhl_psend = [&](std::uint32_t src, std::uint32_t slot) {
#if defined(BHL_ACMPS) && defined(BHL_LRACMP) && defined(BHL_LRASTR)
        {
            std::uint32_t v[32];
            volatile std::uint32_t* ps = reinterpret_cast<volatile std::uint32_t*>(src);
            for (std::uint32_t r = 0; r < 16; ++r) {
                v[r] = ps[r * 16];
                v[16 + r] = ps[512 + r * 16];
            }
            for (std::uint32_t r = 0; r < 32; ++r) {
                ps[(r / 8) * 16 + 2 * (r % 8)] = v[r];
            }
            asm volatile("fence" ::: "memory");
            const std::uint32_t q = slot % 16;
            const std::uint32_t cpa = f5cp.get_write_ptr() + (slot / 16) * acc_bytes + ((q / 4) * 256 + (q % 4) * 64) * 4;
            noc.async_write(CoreLocalMem<std::uint32_t>(src), UnicastEndpoint{}, 256, {.offset_bytes = 0}, {.noc_x = comb_x, .noc_y = comb_y, .addr = cpa});
        }
#elif defined(BHL_FIND) && defined(BHL_LRASTR)
        reinterpret_cast<volatile std::uint32_t*>(src + bhl_fw)[0] = 1u;
        asm volatile("fence" ::: "memory");
        bhl_put(src, comb_x, comb_y, rsum.get_write_ptr() + slot * acc_bytes, bhl_pcol);
        noc.async_write_barrier();
        return;
#else
        bhl_put(src, comb_x, comb_y, rsum.get_write_ptr() + slot * acc_bytes, bhl_pcol);
#endif
#if defined(BHL_PFUSE) && defined(BHL_LRASTR)
        noc.async_writes_flushed();
#else
        noc.async_write_barrier();
#endif
#ifdef BHL_LRASTR
        bhl_flag(bhl_fb + 16 * slot);
        noc.async_write_barrier();
#endif
    };

    for (std::uint32_t r = 0; r < num_rows; ++r) {
#if defined(BHL_LRASTR) || defined(BHL_LRMSM)
        if (is_comb) {
            volatile std::uint32_t* fl = reinterpret_cast<volatile std::uint32_t*>(bhl_fb);
            for (std::uint32_t i = 0; i < (2 * acc_bytes - 16) / 4; i += 4) {
                fl[i] = 0u;
            }
#ifdef BHL_FIND
            for (std::uint32_t k = 0; k < nchunk; ++k) {
                reinterpret_cast<volatile std::uint32_t*>(rsum.get_write_ptr() + k * acc_bytes + bhl_fw)[0] = 0u;
            }
#endif
            asm volatile("fence" ::: "memory");
        }
#endif
#ifdef BHL_LRMSM
        if (is_comb) {
            sem_rmax2.set(1);
            bhl_mcast(noc, CoreLocalMem<std::uint32_t>(bhl_fsrc), 0, 0, x0, y0, x1, y1, gx0, gx1, comb_x, comb_y, &sem_rmax2);
            noc.async_write_barrier();
        }
#endif
        {
        BHLZ("W_MAX");
#ifdef BHL_LRTREE
        if (is_lead) {
            rmax.reserve_back(static_cast<uint16_t>(gsz));
        }
        if (is_comb && ngrp > 1) {
            rmax2.reserve_back(static_cast<uint16_t>(ngrp - 1));
        }
        pmax.wait_front(1);
        {
            CoreLocalMem<std::uint32_t> src(pmax.get_read_ptr());
            noc.async_write(
                src,
                UnicastEndpoint{},
                im_bytes,
                {.offset_bytes = 0},
                {.noc_x = lead_x, .noc_y = lead_y, .addr = rmax.get_write_ptr() + gslot * im_bytes});
            noc.async_write_barrier();
            sem_rmax.up(noc, lead_x, lead_y, 1);
            noc.async_atomic_barrier();
        }
        pmax.pop_front(1);
        mcb.reserve_back(1);
        if (is_lead) {
            sem_rmax.wait(gsz);
            sem_rmax.set(0);
            rmax.push_back(static_cast<uint16_t>(gsz));
            if (!is_comb) {
                mout.wait_front(1);
                CoreLocalMem<std::uint32_t> gsrc(mout.get_read_ptr());
                noc.async_write(
                    gsrc,
                    UnicastEndpoint{},
                    im_bytes,
                    {.offset_bytes = 0},
                    {.noc_x = comb_x, .noc_y = comb_y, .addr = rmax2.get_write_ptr() + (gidx - 1) * im_bytes});
                noc.async_write_barrier();
                sem_rmax2.up(noc, comb_x, comb_y, 1);
                noc.async_atomic_barrier();
                mout.pop_front(1);
            }
        }
        if (is_comb) {
            if (ngrp > 1) {
                sem_rmax2.wait(ngrp - 1);
                sem_rmax2.set(0);
                rmax2.push_back(static_cast<uint16_t>(ngrp - 1));
            }
#else
        if (is_comb) {
            rmax.reserve_back(static_cast<uint16_t>(active));
        }
        pmax.wait_front(1);
        {
            CoreLocalMem<std::uint32_t> src(pmax.get_read_ptr());
            noc.async_write(
                src,
                UnicastEndpoint{},
                im_bytes,
                {.offset_bytes = 0},
                {.noc_x = comb_x, .noc_y = comb_y, .addr = rmax.get_write_ptr() + c * im_bytes});
            noc.async_write_barrier();
#ifdef BHL_LRMSM
            sem_rmax2.wait_min(1);
            sem_rmax2.set(0);
            bhl_flag(bhl_fb + acc_bytes + 16 * c);
            noc.async_write_barrier();
#endif
            sem_rmax.up(noc, comb_x, comb_y, 1);
            noc.async_atomic_barrier();
        }
        pmax.pop_front(1);

        mcb.reserve_back(1);
        if (is_comb) {
#ifdef BHL_LRMSM
            for (std::uint32_t i = 0; i < active; ++i) {
                bhl_poll(bhl_fb + acc_bytes + 16 * i);
                rmax.push_back(1);
            }
#endif
            sem_rmax.wait(active);
            sem_rmax.set(0);
#ifndef BHL_LRMSM
            rmax.push_back(static_cast<uint16_t>(active));
#endif
#endif
            mout.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> one(mout.get_read_ptr());
                const std::uint32_t daddr = mcb.get_write_ptr();
#ifdef BHL_MCASTM
                bhl_mcast(noc, one, im_bytes, daddr, x0, y0, x1, y1, gx0, gx1, comb_x, comb_y, nullptr);
            }
#ifdef BHL_FSEM
            noc.async_writes_flushed();
#else
            noc.async_write_barrier();
#endif
            sem_m.set(1);
            bhl_mcast(noc, CoreLocalMem<std::uint32_t>(mcb.get_write_ptr()), 0, 0, x0, y0, x1, y1, gx0, gx1, comb_x, comb_y, &sem_m);
            noc.async_write_barrier();
#elif defined(BHL_RELAYM)
                for (std::uint32_t y = y0; y <= y1; ++y) {
                    noc.async_write(one, UnicastEndpoint{}, im_bytes, {.offset_bytes = 0}, {.noc_x = x0, .noc_y = y, .addr = daddr});
                }
            }
            noc.async_write_barrier();
            for (std::uint32_t y = y0; y <= y1; ++y) {
                sem_m.up(noc, x0, y, 1);
            }
            noc.async_atomic_barrier();
#else
                for (std::uint32_t y = y0; y <= y1; ++y) {
                    for (std::uint32_t x = x0; x <= x1; ++x) {
                        if (x >= gx0 && x < gx1) {
                            continue;
                        }
                        noc.async_write(
                            one,
                            UnicastEndpoint{},
                            im_bytes,
                            {.offset_bytes = 0},
                            {.noc_x = x, .noc_y = y, .addr = daddr});
                    }
                }
            }
            noc.async_write_barrier();
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_m.up(noc, x, y, 1);
                }
            }
            noc.async_atomic_barrier();
#endif
            mout.pop_front(1);
        }
        sem_m.wait_min(1);
        sem_m.set(0);
#ifdef BHL_RELAYM
        {
            const std::uint32_t rw = (x1 - x0 + 1) - (gx1 > gx0 ? gx1 - gx0 : 0);
            if (c % rw == 0) {
                const std::uint32_t my = y0 + c / rw;
                const std::uint32_t a = mcb.get_write_ptr();
                for (std::uint32_t x = x0 + 1; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    noc.async_write(CoreLocalMem<std::uint32_t>(a), UnicastEndpoint{}, im_bytes, {.offset_bytes = 0}, {.noc_x = x, .noc_y = my, .addr = a});
                }
                noc.async_write_barrier();
                for (std::uint32_t x = x0 + 1; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_m.up(noc, x, my, 1);
                }
                noc.async_atomic_barrier();
            }
        }
#endif
        mcb.push_back(1);
        }
        {
        BHLZ("W_ACC");

        if (is_comb) {
            rsum.reserve_back(static_cast<uint16_t>(nchunk));
        }
#if defined(BHL_LRSEG)
        auto bhl_recv = [&]() {
            if (rcv) {
                racc.reserve_back(1);
                sem_acc.wait_min(1);
                sem_acc.set(0);
                racc.push_back(1);
            }
        };
        auto bhl_send = [&]() {
            if (snd) {
                pacc.wait_front(1);
                bhl_put(pacc.get_read_ptr(), next_x, next_y, racc.get_write_ptr(), bhl_ccol);
                noc.async_write_barrier();
                sem_acc.up(noc, next_x, next_y, 1);
                noc.async_atomic_barrier();
                pacc.pop_front(1);
            }
        };
        if (sfirst) {
            bhl_send();
            bhl_recv();
        } else {
            bhl_recv();
            bhl_send();
        }
        for (std::uint32_t k = 0; k < nps; ++k) {
            psum.wait_front(1);
            bhl_psend(psum.get_read_ptr(), (k == 0 ? pk0 : pk1));
            psum.pop_front(1);
        }
        const std::uint32_t bhl_nps = 0;
        if (nps > 0) {
            sem_rsum.up(noc, comb_x, comb_y, 1);
            noc.async_atomic_barrier();
        }
#elif defined(BHL_CCHAIN)
        if (cj > 0) {
            racc.reserve_back(1);
            sem_acc.wait_min(1);
            sem_acc.set(0);
            racc.push_back(1);
        }
        if (cj + 1 < cS) {
            pacc.wait_front(1);
            bhl_put(pacc.get_read_ptr(), next_x, next_y, racc.get_write_ptr(), bhl_ccol);
            noc.async_write_barrier();
            sem_acc.up(noc, next_x, next_y, 1);
            noc.async_atomic_barrier();
            pacc.pop_front(1);
        }
        const std::uint32_t bhl_nps = (cj + 1 == cS) ? nck : 0u;
#else
        const std::uint32_t bhl_nps = nck;
#endif
        for (std::uint32_t k = 0; k < bhl_nps; ++k) {
            psum.wait_front(1);
            bhl_psend(psum.get_read_ptr(), (ck0 + k));
            psum.pop_front(1);
        }
        if (bhl_nps > 0) {
            sem_rsum.up(noc, comb_x, comb_y, 1);
            noc.async_atomic_barrier();
        }
        }
        {
        BHLZ("W_RB");
        rcb.reserve_back(1);
        if (is_comb) {
#if defined(BHL_LRACMP)
            {
                const std::uint32_t ncomp = (nchunk + 15) / 16;
                f5cp.reserve_back(static_cast<uint16_t>(ncomp));
                volatile std::uint32_t* cp = reinterpret_cast<volatile std::uint32_t*>(f5cp.get_write_ptr());
                const std::uint32_t rs = rsum.get_write_ptr();
                for (std::uint32_t k = 0; k < nchunk; ++k) {
#if defined(BHL_FIND) && !defined(BHL_ACMPS)
                    bhl_poll(rs + k * acc_bytes + bhl_fw);
#else
                    bhl_poll(bhl_fb + 16 * k);
#endif
#ifdef BHL_ACMPS
                    continue;
#endif
                    volatile std::uint32_t* ps = reinterpret_cast<volatile std::uint32_t*>(rs + k * acc_bytes);
                    volatile std::uint32_t* d = cp + (k / 16) * (acc_bytes / 4) + ((k % 16) / 4) * 256 + (k % 4) * 64;
                    for (std::uint32_t r = 0; r < 32; ++r) {
                        d[(r / 8) * 16 + 2 * (r % 8)] = r < 16 ? ps[r * 16] : ps[512 + (r - 16) * 16];
                    }
                }
                asm volatile("fence" ::: "memory");
                f5cp.push_back(static_cast<uint16_t>(ncomp));
                sem_rsum.wait(nsend);
                sem_rsum.set(0);
                f5co.wait_front(1);
                const std::uint32_t j = nchunk - 1;
                volatile std::uint32_t* res = reinterpret_cast<volatile std::uint32_t*>(f5co.get_read_ptr()) + ((j % 16) / 4) * 256 + (j % 4) * 64;
                volatile std::uint32_t* out = reinterpret_cast<volatile std::uint32_t*>(rs);
                for (std::uint32_t r = 0; r < 32; ++r) {
                    out[r < 16 ? r * 16 : 512 + (r - 16) * 16] = res[(r / 8) * 16 + 2 * (r % 8)];
                }
                asm volatile("fence" ::: "memory");
                f5co.pop_front(1);
                rsum.push_back(1);
            }
#else
#ifdef BHL_LRASTR
            [[maybe_unused]] const std::uint32_t bhl_rs0 = rsum.get_write_ptr();
            for (std::uint32_t k = 0; k < nchunk; ++k) {
#ifdef BHL_FIND
                bhl_poll(bhl_rs0 + k * acc_bytes + bhl_fw);
#else
                bhl_poll(bhl_fb + 16 * k);
#endif
                rsum.push_back(1);
            }
#endif
            sem_rsum.wait(nsend);
            sem_rsum.set(0);
#ifndef BHL_LRASTR
            rsum.push_back(static_cast<uint16_t>(nchunk));
#endif
#endif
            rout.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> one(rout.get_read_ptr());
                const std::uint32_t daddr = rcb.get_write_ptr();
#ifdef BHL_MCASTR
                bhl_mcast(noc, one, acc_bytes, daddr, x0, y0, x1, y1, gx0, gx1, comb_x, comb_y, nullptr);
            }
#ifdef BHL_FSEM
            noc.async_writes_flushed();
#else
            noc.async_write_barrier();
#endif
            sem_r.set(1);
            bhl_mcast(noc, CoreLocalMem<std::uint32_t>(rcb.get_write_ptr()), 0, 0, x0, y0, x1, y1, gx0, gx1, comb_x, comb_y, &sem_r);
            noc.async_write_barrier();
#elif defined(BHL_RELAYR)
                for (std::uint32_t y = y0; y <= y1; ++y) {
                    noc.async_write(one, UnicastEndpoint{}, acc_bytes, {.offset_bytes = 0}, {.noc_x = x0, .noc_y = y, .addr = daddr});
                }
            }
            noc.async_write_barrier();
            for (std::uint32_t y = y0; y <= y1; ++y) {
                sem_r.up(noc, x0, y, 1);
            }
            noc.async_atomic_barrier();
#else
                for (std::uint32_t y = y0; y <= y1; ++y) {
                    for (std::uint32_t x = x0; x <= x1; ++x) {
                        if (x >= gx0 && x < gx1) {
                            continue;
                        }
                        noc.async_write(
                            one,
                            UnicastEndpoint{},
                            acc_bytes,
                            {.offset_bytes = 0},
                            {.noc_x = x, .noc_y = y, .addr = daddr});
                    }
                }
            }
            noc.async_write_barrier();
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_r.up(noc, x, y, 1);
                }
            }
            noc.async_atomic_barrier();
#endif
            rout.pop_front(1);
        }
        sem_r.wait_min(1);
        sem_r.set(0);
#ifdef BHL_RELAYR
        {
            const std::uint32_t rw = (x1 - x0 + 1) - (gx1 > gx0 ? gx1 - gx0 : 0);
            if (c % rw == 0) {
                const std::uint32_t my = y0 + c / rw;
                const std::uint32_t a = rcb.get_write_ptr();
                for (std::uint32_t x = x0 + 1; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    noc.async_write(CoreLocalMem<std::uint32_t>(a), UnicastEndpoint{}, acc_bytes, {.offset_bytes = 0}, {.noc_x = x, .noc_y = my, .addr = a});
                }
                noc.async_write_barrier();
                for (std::uint32_t x = x0 + 1; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_r.up(noc, x, my, 1);
                }
                noc.async_atomic_barrier();
            }
        }
#endif
        rcb.push_back(1);
        }
        BHLZ("W_OUT");

        std::uint32_t tile = (row0 + r * rstride) * Wt + t0;
        for (std::uint32_t j = 0; j < T; j += blk) {
            const std::uint32_t rem = (j + blk > T) ? (T - j) : blk;
            out0.wait_front(static_cast<uint16_t>(rem));
#if defined(BHL_WROT) || defined(BHL_WPOST)
            for (std::uint32_t i = 0; i < rem; ++i) {
#ifdef BHL_WROT
                const std::uint32_t q = (i + c) % rem;
#else
                const std::uint32_t q = i;
#endif
#ifdef BHL_WPOST
                noc.async_write<NocOptions::POSTED>(out0, dst, out_bytes, {.offset_bytes = q * out_bytes}, {.page_id = tile + q});
#else
                noc.async_write(out0, dst, out_bytes, {.offset_bytes = q * out_bytes}, {.page_id = tile + q});
#endif
            }
            tile += rem;
#else
            std::uint32_t off = 0;
            for (std::uint32_t i = 0; i < rem; ++i) {
#ifdef BHL_WDNOC
                if (i & 1u) {
                    noc_alt.async_write(out0, dst, out_bytes, {.offset_bytes = off}, {.page_id = tile});
                } else {
                    noc.async_write(out0, dst, out_bytes, {.offset_bytes = off}, {.page_id = tile});
                }
#else
                noc.async_write(out0, dst, out_bytes, {.offset_bytes = off}, {.page_id = tile});
#endif
                ++tile;
                off += out_bytes;
            }
#ifdef BHL_WDNOC
            noc_alt.async_write_barrier();
#endif
#endif
#if defined(BHL_WPOST)
            noc.async_writes_flushed<NocOptions::POSTED>();
#elif defined(BHL_WFLUSH)
            noc.async_writes_flushed();
#else
            noc.async_write_barrier();
#endif
            out0.pop_front(static_cast<uint16_t>(rem));
        }
    }
#ifdef BHL_WFLUSH
    noc.async_write_barrier();
#endif
}
