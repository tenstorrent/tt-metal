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
#ifdef BHF_PROF
#include "api/debug/kernel_profiler.hpp"
#define BHFZ(n) DeviceZoneScopedN(n)
#else
#define BHFZ(n)
#endif

#if defined(BHF_MAXRISC) || defined(BHF_MAXAG) || defined(BHF_MAXAGC)
inline std::uint32_t bhf_key(std::uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
inline std::uint32_t bhf_col(std::uint32_t i) { return i < 16 ? i * 16 : 512 + (i - 16) * 16; }
inline void bhf_colmax(std::uint32_t dst, std::uint32_t src, std::uint32_t n, std::uint32_t stride, bool compact) {
    volatile std::uint32_t* d = reinterpret_cast<volatile std::uint32_t*>(dst);
    for (std::uint32_t i = 0; i < 32; ++i) {
        const std::uint32_t si = compact ? i : bhf_col(i);
        std::uint32_t best = reinterpret_cast<volatile std::uint32_t*>(src)[si];
        std::uint32_t bk = bhf_key(best);
        for (std::uint32_t k = 1; k < n; ++k) {
            const std::uint32_t v = reinterpret_cast<volatile std::uint32_t*>(src + k * stride)[si];
            const std::uint32_t vk = bhf_key(v);
            if (vk > bk) {
                bk = vk;
                best = v;
            }
        }
        d[bhf_col(i)] = best;
    }
    asm volatile("fence" ::: "memory");
}
#endif
#if defined(BHF_MCASTM) || defined(BHF_MCASTR)
template <typename Src>
inline std::uint32_t bhf_mcast(
    const Noc& noc, const Src& src, std::uint32_t bytes, std::uint32_t addr, std::uint32_t x0, std::uint32_t y0,
    std::uint32_t x1, std::uint32_t y1, std::uint32_t gx0, std::uint32_t gx1, std::uint32_t sx, bool sem_only,
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
        const bool incl = sx >= xs && sx <= xe;
        if (noc.get_noc_id() == 1) {
            std::uint32_t t = xs; xs = xe; xe = t;
            t = ys; ys = ye; ye = t;
        }
        if (!sem_only) {
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
    return nr;
}
#endif

void kernel_main() {
    const std::uint32_t c = get_arg(args::c);
    const std::uint32_t C = get_arg(args::C);
    [[maybe_unused]] const std::uint32_t comb_x = get_arg(args::comb_x);
    [[maybe_unused]] const std::uint32_t comb_y = get_arg(args::comb_y);
    [[maybe_unused]] const std::uint32_t next_x = get_arg(args::next_x);
    [[maybe_unused]] const std::uint32_t next_y = get_arg(args::next_y);
    [[maybe_unused]] const std::uint32_t x0 = get_arg(args::x0);
    [[maybe_unused]] const std::uint32_t y0 = get_arg(args::y0);
    [[maybe_unused]] const std::uint32_t x1 = get_arg(args::x1);
    [[maybe_unused]] const std::uint32_t y1 = get_arg(args::y1);
    const std::uint32_t T = get_arg(args::T);
    const std::uint32_t t0 = get_arg(args::t0);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const std::uint32_t blk = get_arg(args::blk);
    const std::uint32_t row0 = get_arg(args::row0);
    const std::uint32_t rstride = get_arg(args::rstride);
    [[maybe_unused]] const std::uint32_t gx0 = get_arg(args::gx0);
    [[maybe_unused]] const std::uint32_t gx1 = get_arg(args::gx1);
    [[maybe_unused]] const std::uint32_t gm = get_arg(args::gm);
    [[maybe_unused]] const std::uint32_t gi = get_arg(args::gi);
    [[maybe_unused]] const std::uint32_t gt = get_arg(args::gt);
    [[maybe_unused]] const std::uint32_t th = get_arg(args::th);
    [[maybe_unused]] const std::uint32_t tx = get_arg(args::tx);
    [[maybe_unused]] const std::uint32_t ty = get_arg(args::ty);
    const std::uint32_t ts = get_arg(args::ts);
    [[maybe_unused]] const std::uint32_t am = get_arg(args::am);
    [[maybe_unused]] const std::uint32_t ax2 = get_arg(args::ax2);
    [[maybe_unused]] const std::uint32_t ay2 = get_arg(args::ay2);
    [[maybe_unused]] const std::uint32_t mx = get_arg(args::mx);
    [[maybe_unused]] const std::uint32_t my = get_arg(args::my);
    [[maybe_unused]] const std::uint32_t bi = get_arg(args::bi);
    [[maybe_unused]] const std::uint32_t nb = get_arg(args::nb);
    [[maybe_unused]] const std::uint32_t bl = get_arg(args::bl);
    [[maybe_unused]] const std::uint32_t acx0 = get_arg(args::acx0);
    [[maybe_unused]] const std::uint32_t acy0 = get_arg(args::acy0);
    [[maybe_unused]] const std::uint32_t acx1 = get_arg(args::acx1);
    [[maybe_unused]] const std::uint32_t acy1 = get_arg(args::acy1);
    [[maybe_unused]] const std::uint32_t acx2 = get_arg(args::acx2);
    [[maybe_unused]] const std::uint32_t acy2 = get_arg(args::acy2);
    [[maybe_unused]] const std::uint32_t acx3 = get_arg(args::acx3);
    [[maybe_unused]] const std::uint32_t acy3 = get_arg(args::acy3);
    [[maybe_unused]] const std::uint32_t acx4 = get_arg(args::acx4);
    [[maybe_unused]] const std::uint32_t acy4 = get_arg(args::acy4);
    [[maybe_unused]] const std::uint32_t acx5 = get_arg(args::acx5);
    [[maybe_unused]] const std::uint32_t acy5 = get_arg(args::acy5);
    [[maybe_unused]] const std::uint32_t acx6 = get_arg(args::acx6);
    [[maybe_unused]] const std::uint32_t acy6 = get_arg(args::acy6);
    [[maybe_unused]] const std::uint32_t acx7 = get_arg(args::acx7);
    [[maybe_unused]] const std::uint32_t acy7 = get_arg(args::acy7);
#ifdef BHF_BLOCKACC
    const std::uint32_t acx[8] = {acx0, acx1, acx2, acx3, acx4, acx5, acx6, acx7};
    const std::uint32_t acy[8] = {acy0, acy1, acy2, acy3, acy4, acy5, acy6, acy7};
    const std::uint32_t FLB = bl;
#else
    [[maybe_unused]] const std::uint32_t FLB = Wt;
#endif
    if ((T == 0 && am != 2) || num_rows == 0) {
        return;
    }
    [[maybe_unused]] const bool is_comb = c == 0;
#ifdef BHF_GROUP
    [[maybe_unused]] const bool is_member = gi != 0;
    const bool is_last = !is_member && c + gm >= C;
#else
    [[maybe_unused]] const bool is_member = false;
    const bool is_last = c + 1 == C;
#endif

    const Noc noc;
#ifdef BHF_DUALNOC
    const Noc noc_alt(static_cast<uint8_t>(1 - noc.get_noc_id()));
#endif
    DataflowBuffer out0(dfb::out0);
    DataflowBuffer pmax(dfb::pmax);
    DataflowBuffer rmax(dfb::rmax);
    DataflowBuffer mout(dfb::mout);
    DataflowBuffer mcb(dfb::mcb);
    DataflowBuffer pacc(dfb::pacc);
    DataflowBuffer racc(dfb::racc);
    DataflowBuffer rout(dfb::rout);
    DataflowBuffer rcb(dfb::rcb);
    [[maybe_unused]] DataflowBuffer gexp(dfb::gexp);
    Semaphore sem_rmax(sem::rmax_sem);
    Semaphore sem_m(sem::m_sem);
    Semaphore sem_acc(sem::acc_sem);
    Semaphore sem_r(sem::r_sem);
    [[maybe_unused]] const auto dst = TensorAccessor(tensor::dst);
    [[maybe_unused]] const std::uint32_t out_bytes = out0.get_entry_size();
    [[maybe_unused]] const std::uint32_t im_bytes = pmax.get_entry_size();
    [[maybe_unused]] const std::uint32_t acc_bytes = pacc.get_entry_size();

#ifndef BHF_NACCV
#define BHF_NACCV 1
#endif
#ifdef BHF_STREAM
    if (am == 2) {
        BHFZ("A_POLL");
        const std::uint32_t eb = gexp.get_entry_size();
        const std::uint32_t gbase = gexp.get_write_ptr();
        volatile std::uint32_t* fl = reinterpret_cast<volatile std::uint32_t*>(gbase + FLB * eb);
#ifdef BHF_ACCSEQ
        const std::uint32_t tb = gi == 0 ? 0u : th;
        const std::uint32_t te = gi == 0 ? th : Wt;
        if (gi != 0) {
            racc.reserve_back(1);
            sem_acc.wait_min(1);
            sem_acc.set(0);
            racc.push_back(1);
        }
#elif defined(BHF_BLOCKACC)
        const std::uint32_t tb = 0, te = gt;
#else
        const std::uint32_t tb = 0, te = Wt;
#endif
        gexp.reserve_back(static_cast<uint16_t>(te - tb));
        for (std::uint32_t t = tb; t < te; ++t) {
            while (true) {
                invalidate_l1_cache();
                if (fl[t * 4] == 0xF5A50000u + t) {
                    break;
                }
            }
            fl[t * 4] = 0;
            gexp.push_back(1);
        }
#ifdef BHF_BLOCKACC
        if (bi != 0) {
            pacc.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> src(pacc.get_read_ptr());
                noc.async_write(src, UnicastEndpoint{}, acc_bytes, {.offset_bytes = 0},
                                {.noc_x = next_x, .noc_y = next_y, .addr = racc.get_write_ptr() + (bi - 1) * acc_bytes});
                noc.async_writes_flushed();
                fl[(128 + bi) * 4] = 0xF5C50000u + bi;
                asm volatile("fence" ::: "memory");
                CoreLocalMem<std::uint32_t> fsrc(gbase + FLB * eb + (128 + bi) * 16);
                noc.async_write(fsrc, UnicastEndpoint{}, 16, {.offset_bytes = 0},
                                {.noc_x = next_x, .noc_y = next_y, .addr = gbase + FLB * eb + (128 + bi) * 16});
            }
            noc.async_write_barrier();
            pacc.pop_front(1);
            return;
        }
        if (nb > 1) {
            racc.reserve_back(static_cast<uint16_t>(nb - 1));
            for (std::uint32_t k = 1; k < nb; ++k) {
                while (true) {
                    invalidate_l1_cache();
                    if (fl[(128 + k) * 4] == 0xF5C50000u + k) {
                        break;
                    }
                }
                fl[(128 + k) * 4] = 0;
                racc.push_back(1);
            }
        }
#endif
#ifdef BHF_ACCSEQ
        if (gi == 0) {
            pacc.wait_front(1);
            CoreLocalMem<std::uint32_t> src(pacc.get_read_ptr());
            noc.async_write(src, UnicastEndpoint{}, acc_bytes, {.offset_bytes = 0},
                            {.noc_x = next_x, .noc_y = next_y, .addr = racc.get_write_ptr()});
            noc.async_write_barrier();
            sem_acc.up(noc, next_x, next_y, 1);
            noc.async_atomic_barrier();
            pacc.pop_front(1);
            return;
        }
#endif
        rout.wait_front(1);
        {
#if BHF_NACCV == 2
            const std::uint32_t off = gi == 0 ? 0u : acc_bytes / 2;
            const std::uint32_t nb = acc_bytes / 4;
#else
            const std::uint32_t off = 0u;
            const std::uint32_t nb = acc_bytes;
#endif
            CoreLocalMem<std::uint32_t> one(rout.get_read_ptr());
            const std::uint32_t daddr = rcb.get_write_ptr();
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    noc.async_write(one, UnicastEndpoint{}, nb, {.offset_bytes = off}, {.noc_x = x, .noc_y = y, .addr = daddr + off});
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
        rout.pop_front(1);
        return;
    }
#endif
    for (std::uint32_t r = 0; r < num_rows; ++r) {
        {
        BHFZ("W_MAX");
#if defined(NUMERIC_STABLE) && defined(BHF_ABL_NOMAXX)
        pmax.wait_front(1);
        pmax.pop_front(1);
        mcb.reserve_back(1);
        mcb.push_back(1);
#elif defined(NUMERIC_STABLE) && (defined(BHF_MAXAG) || defined(BHF_MAXAGC))
        rmax.reserve_back(static_cast<uint16_t>(C));
        pmax.wait_front(1);
        {
            const std::uint32_t rbase = rmax.get_write_ptr();
#ifdef BHF_MAXAGC
            const std::uint32_t slot = 128;
            {
                invalidate_l1_cache();
                volatile std::uint32_t* pp = reinterpret_cast<volatile std::uint32_t*>(pmax.get_read_ptr());
                std::uint32_t tmp[32];
                for (std::uint32_t i = 0; i < 32; ++i) {
                    tmp[i] = pp[bhf_col(i)];
                }
                for (std::uint32_t i = 0; i < 32; ++i) {
                    pp[i] = tmp[i];
                }
                asm volatile("fence" ::: "memory");
            }
#else
            const std::uint32_t slot = im_bytes;
#endif
            CoreLocalMem<std::uint32_t> src(pmax.get_read_ptr());
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    noc.async_write(
                        src, UnicastEndpoint{}, slot, {.offset_bytes = 0},
                        {.noc_x = x, .noc_y = y, .addr = rbase + c * slot});
                }
            }
#ifdef BHF_FLUSH
            noc.async_writes_flushed();
#else
            noc.async_write_barrier();
#endif
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_rmax.up(noc, x, y, 1);
                }
            }
#ifndef BHF_FLUSH
            noc.async_atomic_barrier();
#endif
            pmax.pop_front(1);
            mcb.reserve_back(1);
            sem_rmax.wait(C);
            sem_rmax.set(0);
            invalidate_l1_cache();
#ifdef BHF_MAXAGC
            bhf_colmax(mcb.get_write_ptr(), rbase, C, slot, true);
#else
            bhf_colmax(mcb.get_write_ptr(), rbase, C, slot, false);
#endif
            mcb.push_back(1);
        }
#elif defined(NUMERIC_STABLE)
#ifdef BHF_TREE2
        const bool bhf_up = c >= th;
        const std::uint32_t bhf_dx = bhf_up ? tx : comb_x, bhf_dy = bhf_up ? ty : comb_y, bhf_slot = bhf_up ? c - th : c;
        if (is_comb || c == th) {
            rmax.reserve_back(static_cast<uint16_t>(is_comb ? th + 1 : C - th));
        }
#else
        const std::uint32_t bhf_dx = comb_x, bhf_dy = comb_y, bhf_slot = c;
        if (is_comb) {
            rmax.reserve_back(static_cast<uint16_t>(C));
        }
#endif
        pmax.wait_front(1);
        {
            CoreLocalMem<std::uint32_t> src(pmax.get_read_ptr());
            noc.async_write(
                src,
                UnicastEndpoint{},
                im_bytes,
                {.offset_bytes = 0},
                {.noc_x = bhf_dx, .noc_y = bhf_dy, .addr = rmax.get_write_ptr() + bhf_slot * im_bytes});
#ifdef BHF_FLUSH
            noc.async_writes_flushed();
#else
            noc.async_write_barrier();
#endif
#ifdef BHF_COMBSTREAM
            {
                const std::uint32_t mfl = gexp.get_write_ptr() + (Wt + 1) * gexp.get_entry_size();
                volatile std::uint32_t* mf = reinterpret_cast<volatile std::uint32_t*>(mfl);
                mf[bhf_slot * 4] = 0xF5B50000u + bhf_slot;
                asm volatile("fence" ::: "memory");
                CoreLocalMem<std::uint32_t> msrc(mfl + bhf_slot * 16);
                noc.async_writes_flushed();
                noc.async_write(msrc, UnicastEndpoint{}, 16, {.offset_bytes = 0},
                                {.noc_x = bhf_dx, .noc_y = bhf_dy, .addr = mfl + bhf_slot * 16});
                noc.async_writes_flushed();
            }
#else
            sem_rmax.up(noc, bhf_dx, bhf_dy, 1);
#ifndef BHF_FLUSH
            noc.async_atomic_barrier();
#endif
#endif
        }
        pmax.pop_front(1);
#ifdef BHF_TREE2
        if (c == th) {
            sem_rmax.wait(C - th);
            sem_rmax.set(0);
            rmax.push_back(static_cast<uint16_t>(C - th));
            mout.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> one(mout.get_read_ptr());
                noc.async_write(one, UnicastEndpoint{}, im_bytes, {.offset_bytes = 0},
                                {.noc_x = comb_x, .noc_y = comb_y, .addr = rmax.get_write_ptr() + th * im_bytes});
            }
            noc.async_write_barrier();
            sem_rmax.up(noc, comb_x, comb_y, 1);
            noc.async_atomic_barrier();
            mout.pop_front(1);
        }
#endif

        mcb.reserve_back(1);
        if (is_comb) {
#ifdef BHF_TREE2
            sem_rmax.wait(th + 1);
            sem_rmax.set(0);
            rmax.push_back(static_cast<uint16_t>(th + 1));
            mout.wait_front(1);
#elif defined(BHF_COMBSTREAM)
            {
                volatile std::uint32_t* mf =
                    reinterpret_cast<volatile std::uint32_t*>(gexp.get_write_ptr() + (Wt + 1) * gexp.get_entry_size());
                for (std::uint32_t i = 0; i < C; ++i) {
                    while (true) {
                        invalidate_l1_cache();
                        if (mf[i * 4] == 0xF5B50000u + i) {
                            break;
                        }
                    }
                    mf[i * 4] = 0;
                    rmax.push_back(1);
                }
            }
#else
            sem_rmax.wait(C);
            sem_rmax.set(0);
#endif
#if defined(BHF_MAXRISC) && !defined(BHF_TREE2)
            invalidate_l1_cache();
            bhf_colmax(rmax.get_write_ptr(), rmax.get_write_ptr(), C, im_bytes, false);
#elif !defined(BHF_TREE2)
#ifndef BHF_COMBSTREAM
            rmax.push_back(static_cast<uint16_t>(C));
#endif
            mout.wait_front(1);
#endif
            {
#if defined(BHF_MAXRISC) && !defined(BHF_TREE2)
                CoreLocalMem<std::uint32_t> one(rmax.get_write_ptr());
#else
                CoreLocalMem<std::uint32_t> one(mout.get_read_ptr());
#endif
                const std::uint32_t daddr = mcb.get_write_ptr();
#ifdef BHF_MCASTM
                bhf_mcast(noc, one, im_bytes, daddr, x0, y0, x1, y1, gx0, gx1, x0, false, nullptr);
            }
            noc.async_write_barrier();
            sem_m.set(1);
            bhf_mcast(noc, CoreLocalMem<std::uint32_t>(mcb.get_write_ptr()), 0, 0, x0, y0, x1, y1, gx0, gx1, x0, true, &sem_m);
            noc.async_write_barrier();
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
#ifdef BHF_FLUSH
            noc.async_writes_flushed();
#else
            noc.async_write_barrier();
#endif
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_m.up(noc, x, y, 1);
                }
            }
#ifndef BHF_FLUSH
            noc.async_atomic_barrier();
#endif
#endif
#if !defined(BHF_MAXRISC) || defined(BHF_TREE2)
            mout.pop_front(1);
#endif
        }
        sem_m.wait_min(1);
        sem_m.set(0);
        mcb.push_back(1);
#endif
        }

#ifdef BHF_STREAM
        if (is_member) {
            BHFZ("P_FWD");
            const std::uint32_t eb = gexp.get_entry_size();
            const std::uint32_t gbase = gexp.get_write_ptr();
            volatile std::uint32_t* myfl = reinterpret_cast<volatile std::uint32_t*>(gbase + FLB * eb);
#ifdef BHF_DIRECT
            gexp.reserve_back(static_cast<uint16_t>(T));
#endif
            for (std::uint32_t j = 0; j < T; ++j) {
                const std::uint32_t tg = t0 + j * ts;
#ifdef BHF_BLOCKACC
                const std::uint32_t fbk = tg / bl;
                const std::uint32_t slot = tg - fbk * bl;
                const std::uint32_t ux = acx[fbk * BHF_NACCV], uy = acy[fbk * BHF_NACCV];
                [[maybe_unused]] const std::uint32_t lx = acx[fbk * BHF_NACCV + BHF_NACCV - 1], ly = acy[fbk * BHF_NACCV + BHF_NACCV - 1];
#else
                const std::uint32_t slot = tg;
                const std::uint32_t ux = next_x, uy = next_y;
                [[maybe_unused]] const std::uint32_t lx = ax2, ly = ay2;
#endif
                pacc.wait_front(1);
#ifdef BHF_DIRECT
                {
                    CoreLocalMem<std::uint32_t> lsrc(pacc.get_read_ptr());
                    noc.async_write(lsrc, UnicastEndpoint{}, eb, {.offset_bytes = 0}, {.noc_x = mx, .noc_y = my, .addr = gbase + j * eb});
                }
#endif
                myfl[slot * 4] = 0xF5A50000u + slot;
                asm volatile("fence" ::: "memory");
                CoreLocalMem<std::uint32_t> src(pacc.get_read_ptr());
                CoreLocalMem<std::uint32_t> fsrc(gbase + FLB * eb + slot * 16);
#if BHF_NACCV == 2
                noc.async_write(src, UnicastEndpoint{}, eb / 2, {.offset_bytes = 0},
                                {.noc_x = ux, .noc_y = uy, .addr = gbase + slot * eb});
                noc.async_write(src, UnicastEndpoint{}, eb / 2, {.offset_bytes = eb / 2},
                                {.noc_x = lx, .noc_y = ly, .addr = gbase + slot * eb + eb / 2});
                noc.async_writes_flushed();
                noc.async_write(fsrc, UnicastEndpoint{}, 16, {.offset_bytes = 0},
                                {.noc_x = ux, .noc_y = uy, .addr = gbase + FLB * eb + slot * 16});
                noc.async_write(fsrc, UnicastEndpoint{}, 16, {.offset_bytes = 0},
                                {.noc_x = lx, .noc_y = ly, .addr = gbase + FLB * eb + slot * 16});
#elif defined(BHF_BLOCKACC)
                noc.async_write(src, UnicastEndpoint{}, eb, {.offset_bytes = 0}, {.noc_x = ux, .noc_y = uy, .addr = gbase + slot * eb});
                noc.async_writes_flushed();
                noc.async_write(fsrc, UnicastEndpoint{}, 16, {.offset_bytes = 0},
                                {.noc_x = ux, .noc_y = uy, .addr = gbase + FLB * eb + slot * 16});
#else
#ifdef BHF_ACCSEQ
                const bool bhf_b = tg >= th;
                const std::uint32_t dx = bhf_b ? ax2 : next_x, dy = bhf_b ? ay2 : next_y;
#else
                const std::uint32_t dx = next_x, dy = next_y;
#endif
#ifdef BHF_DUALNOC
                const Noc& nn = (tg & 1u) ? noc_alt : noc;
#else
                const Noc& nn = noc;
#endif
                nn.async_write(src, UnicastEndpoint{}, eb, {.offset_bytes = 0}, {.noc_x = dx, .noc_y = dy, .addr = gbase + tg * eb});
                nn.async_writes_flushed();
                nn.async_write(fsrc, UnicastEndpoint{}, 16, {.offset_bytes = 0},
                               {.noc_x = dx, .noc_y = dy, .addr = gbase + Wt * eb + tg * 16});
                nn.async_writes_flushed();
#endif
                noc.async_writes_flushed();
                pacc.pop_front(1);
            }
            noc.async_write_barrier();
#ifdef BHF_DIRECT
            gexp.push_back(static_cast<uint16_t>(T));
#endif
#ifdef BHF_DUALNOC
            noc_alt.async_write_barrier();
#endif
        }
#elif defined(BHF_GROUP)
        if (is_member) {
            const std::uint32_t gaddr = gexp.get_write_ptr() + gt * acc_bytes;
            for (std::uint32_t j = 0; j < T; ++j) {
                pacc.wait_front(1);
                CoreLocalMem<std::uint32_t> src(pacc.get_read_ptr());
                noc.async_write(
                    src, UnicastEndpoint{}, acc_bytes, {.offset_bytes = 0},
                    {.noc_x = next_x, .noc_y = next_y, .addr = gaddr + j * acc_bytes});
                noc.async_writes_flushed();
                pacc.pop_front(1);
            }
            noc.async_write_barrier();
            sem_rmax.up(noc, next_x, next_y, 1);
            noc.async_atomic_barrier();
#ifndef BHF_GROUPMUL
            continue;
#endif
        }
#endif
        if (!is_member) {
        BHFZ("W_ACC");
#ifdef BHF_ABL_NOSUM
        if (!is_last) {
            pacc.wait_front(1);
            pacc.pop_front(1);
        }
#else
        if (c > 0) {
            BHFZ("W_RWAIT");
            racc.reserve_back(1);
            sem_acc.wait_min(1);
            sem_acc.set(0);
            racc.push_back(1);
        }
#ifdef BHF_GROUP
        if (gt != 0) {
            gexp.reserve_back(static_cast<uint16_t>(gt));
            sem_rmax.wait(gm - 1);
            sem_rmax.set(0);
            gexp.push_back(static_cast<uint16_t>(gt));
        }
#endif
        if (!is_last) {
            {
            BHFZ("W_PWAIT");
            pacc.wait_front(1);
            }
            BHFZ("W_SEND");
            CoreLocalMem<std::uint32_t> src(pacc.get_read_ptr());
#ifdef BHF_ACCHALF
            noc.async_write(
                src, UnicastEndpoint{}, acc_bytes / 4, {.offset_bytes = 0},
                {.noc_x = next_x, .noc_y = next_y, .addr = racc.get_write_ptr()});
            noc.async_write(
                src, UnicastEndpoint{}, acc_bytes / 4, {.offset_bytes = acc_bytes / 2},
                {.noc_x = next_x, .noc_y = next_y, .addr = racc.get_write_ptr() + acc_bytes / 2});
#else
            noc.async_write(
                src,
                UnicastEndpoint{},
                acc_bytes,
                {.offset_bytes = 0},
                {.noc_x = next_x, .noc_y = next_y, .addr = racc.get_write_ptr()});
#endif
#ifdef BHF_FLUSH
            noc.async_writes_flushed();
#else
            noc.async_write_barrier();
#endif
            sem_acc.up(noc, next_x, next_y, 1);
#ifndef BHF_FLUSH
            noc.async_atomic_barrier();
#endif
            pacc.pop_front(1);
        }
#endif
        }

        {
        BHFZ("W_RB");
        rcb.reserve_back(1);
#ifdef BHF_ABL_NORB
        if (is_last) {
            rout.wait_front(1);
            rout.pop_front(1);
        }
#else
        if (is_last) {
            rout.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> one(rout.get_read_ptr());
                const std::uint32_t daddr = rcb.get_write_ptr();
#ifdef BHF_MCASTR
                bhf_mcast(noc, one, acc_bytes, daddr, x0, y0, x1, y1, gx0, gx1, x1, false, nullptr);
            }
            noc.async_write_barrier();
            sem_r.set(1);
            bhf_mcast(noc, CoreLocalMem<std::uint32_t>(rcb.get_write_ptr()), 0, 0, x0, y0, x1, y1, gx0, gx1, x1, true, &sem_r);
            noc.async_write_barrier();
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
#ifdef BHF_FLUSH
            noc.async_writes_flushed();
#else
            noc.async_write_barrier();
#endif
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_r.up(noc, x, y, 1);
                }
            }
#ifndef BHF_FLUSH
            noc.async_atomic_barrier();
#endif
#endif
            rout.pop_front(1);
        }
        sem_r.wait_min(BHF_NACCV);
        sem_r.set(0);
#endif
        rcb.push_back(1);
        }
        BHFZ("W_OUT");

        std::uint32_t tile = (row0 + r * rstride) * Wt + t0;
#if defined(BHF_GROUP) && !defined(BHF_GROUPMUL)
        const std::uint32_t TT = T + gt;
#else
        const std::uint32_t TT = T;
#endif
        for (std::uint32_t j = 0; j < TT; j += blk) {
            const std::uint32_t rem = (j + blk > TT) ? (TT - j) : blk;
            out0.wait_front(static_cast<uint16_t>(rem));
            std::uint32_t off = 0;
            for (std::uint32_t i = 0; i < rem; ++i) {
#ifndef BHF_ABL_NOWR
                noc.async_write(out0, dst, out_bytes, {.offset_bytes = off}, {.page_id = tile});
#endif
                tile += ts;
                off += out_bytes;
            }
            noc.async_writes_flushed();
            out0.pop_front(static_cast<uint16_t>(rem));
        }
    }
    noc.async_write_barrier();
#ifdef BHF_FLUSH
    noc.async_atomic_barrier();
#endif
}
