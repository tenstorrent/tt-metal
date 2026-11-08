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

#if defined(BH_RECIP_MCAST)
template <typename SrcT>
inline bool sm_mcast(const Noc& noc, const SrcT& src, std::uint32_t bytes, std::uint32_t x0, std::uint32_t y0, std::uint32_t x1,
                     std::uint32_t y1, std::uint32_t gx0, std::uint32_t gx1, std::uint32_t addr) {
    if (gx1 > gx0 && x0 < gx1 && x1 >= gx0) {
        return false;
    }
    const std::uint32_t nd = (x1 - x0 + 1) * (y1 - y0 + 1);
    if (noc.get_noc_id() == 0) {
        noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
            src, MulticastEndpoint{}, bytes, nd, {.offset_bytes = 0},
            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = addr});
    } else {
        noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
            src, MulticastEndpoint{}, bytes, nd, {.offset_bytes = 0},
            {.noc_x_start = x1, .noc_y_start = y1, .noc_x_end = x0, .noc_y_end = y0, .addr = addr});
    }
    return true;
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
    if (T == 0 || num_rows == 0) {
        return;
    }
    [[maybe_unused]] const bool is_comb = c == 0;
#if defined(BH_ACC)
    const bool is_last = c == 0;
#else
    const bool is_last = c + 1 == C;
#endif

    const Noc noc;
    DataflowBuffer out0(dfb::out0);
    DataflowBuffer pmax(dfb::pmax);
    DataflowBuffer rmax(dfb::rmax);
    DataflowBuffer mout(dfb::mout);
    DataflowBuffer mcb(dfb::mcb);
    DataflowBuffer pacc(dfb::pacc);
    DataflowBuffer racc(dfb::racc);
    DataflowBuffer rout(dfb::rout);
    DataflowBuffer rcb(dfb::rcb);
    Semaphore sem_rmax(sem::rmax_sem);
    Semaphore sem_m(sem::m_sem);
    Semaphore sem_acc(sem::acc_sem);
    Semaphore sem_r(sem::r_sem);
    [[maybe_unused]] const auto dst = TensorAccessor(tensor::dst);
    [[maybe_unused]] const std::uint32_t out_bytes = out0.get_entry_size();
    [[maybe_unused]] const std::uint32_t im_bytes = pmax.get_entry_size();
#if defined(BH_ACC)
    DataflowBuffer exr(dfb::exr);
    DataflowBuffer exs(dfb::exs);
    DataflowBuffer flg(dfb::flg);
    const std::uint32_t sm_xr = exr.get_write_ptr();
    const std::uint32_t sm_fl = flg.get_write_ptr();
    if (c == 0) {
        for (std::uint32_t i = 0; i < Wt * 4; ++i) {
            reinterpret_cast<volatile std::uint32_t*>(sm_fl)[i] = 0;
        }
    }
    const Noc sm_nb(static_cast<std::uint8_t>(1 - noc_index));
#endif
    [[maybe_unused]] const std::uint32_t acc_bytes = pacc.get_entry_size();

    for (std::uint32_t r = 0; r < num_rows; ++r) {
        {
#if defined(NUMERIC_STABLE)
        if (is_comb) {
            rmax.reserve_back(static_cast<uint16_t>(C));
        }
        pmax.wait_front(1);
        {
            CoreLocalMem<std::uint32_t> src(pmax.get_read_ptr());
            {
                noc.async_write(src, UnicastEndpoint{}, im_bytes, {.offset_bytes = 0}, {.noc_x = comb_x, .noc_y = comb_y, .addr = rmax.get_write_ptr() + c * im_bytes});
                }
            noc.async_writes_flushed();
            sem_rmax.up(noc, comb_x, comb_y, 1);
        }
        pmax.pop_front(1);

        mcb.reserve_back(1);
        if (is_comb) {
            sem_rmax.wait(C);
            sem_rmax.set(0);
            rmax.push_back(static_cast<uint16_t>(C));
            mout.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> one(mout.get_read_ptr());
                const std::uint32_t daddr = mcb.get_write_ptr();
                for (std::uint32_t y = y0; y <= y1; ++y) {
                    for (std::uint32_t x = x0; x <= x1; ++x) {
                        if (x >= gx0 && x < gx1) {
                            continue;
                        }
                        noc.async_write(one, UnicastEndpoint{}, im_bytes, {.offset_bytes = 0}, {.noc_x = x, .noc_y = y, .addr = daddr});
                    }
                }
            }
            noc.async_writes_flushed();
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_m.up(noc, x, y, 1);
                }
            }
            mout.pop_front(1);
        }
        sem_m.wait_min(1);
        sem_m.set(0);
        mcb.push_back(1);
#endif
        }

        {
#if defined(BH_ACC)
        if (c == 0) {
            exr.reserve_back(static_cast<uint16_t>(Wt));
            exr.push_back(static_cast<uint16_t>(T));
            for (std::uint32_t b = T; b < Wt; b += blk) {
                const std::uint32_t rem = (b + blk > Wt) ? (Wt - b) : blk;
                for (std::uint32_t i = 0; i < rem; ++i) {
                    volatile std::uint32_t* f = reinterpret_cast<volatile std::uint32_t*>(sm_fl + (b + i) * 16);
                    while (*f < r + 1) {
                        invalidate_l1_cache();
                    }
                }
                exr.push_back(static_cast<uint16_t>(rem));
            }
        } else {
            for (std::uint32_t j = 0; j < T; j += blk) {
                const std::uint32_t rem = (j + blk > T) ? (T - j) : blk;
                exs.wait_front(static_cast<uint16_t>(rem));
                const std::uint32_t fo = (t0 + j) * 16;
                for (std::uint32_t i = 0; i < rem; ++i) {
                    *reinterpret_cast<volatile std::uint32_t*>(sm_fl + fo + i * 16) = r + 1;
                }
                CoreLocalMem<std::uint32_t> xs(exs.get_read_ptr());
                CoreLocalMem<std::uint32_t> fs(sm_fl + fo);
                const Noc& nn = (c & 1) ? sm_nb : noc;
                nn.async_write(
                    xs,
                    UnicastEndpoint{},
                    rem * im_bytes,
                    {.offset_bytes = 0},
                    {.noc_x = comb_x, .noc_y = comb_y, .addr = sm_xr + (t0 + j) * im_bytes});
                nn.async_writes_flushed();
                nn.async_write(
                    fs, UnicastEndpoint{}, rem * 16, {.offset_bytes = 0}, {.noc_x = comb_x, .noc_y = comb_y, .addr = sm_fl + fo});
                nn.async_writes_flushed();
                exs.pop_front(static_cast<uint16_t>(rem));
            }
        }
#else
        if (c > 0) {
            racc.reserve_back(1);
            sem_acc.wait_min(1);
            sem_acc.set(0);
            racc.push_back(1);
        }
        if (!is_last) {
            {
            pacc.wait_front(1);
            }
            CoreLocalMem<std::uint32_t> src(pacc.get_read_ptr());
            {
                noc.async_write(src, UnicastEndpoint{}, acc_bytes, {.offset_bytes = 0}, {.noc_x = next_x, .noc_y = next_y, .addr = racc.get_write_ptr()});
                }
            noc.async_writes_flushed();
            sem_acc.up(noc, next_x, next_y, 1);
            pacc.pop_front(1);
        }
#endif
        }

        {
        rcb.reserve_back(1);
        if (is_last) {
            rout.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> one(rout.get_read_ptr());
                const std::uint32_t daddr = rcb.get_write_ptr();
#if defined(BH_RECIP_MCAST)
                if (!sm_mcast(noc, one, acc_bytes, x0, y0, x1, y1, gx0, gx1, daddr))
#endif
                for (std::uint32_t y = y0; y <= y1; ++y) {
                    for (std::uint32_t x = x0; x <= x1; ++x) {
                        if (x >= gx0 && x < gx1) {
                            continue;
                        }
#if defined(BH_RECIP_FACES)
                        noc.async_write(one, UnicastEndpoint{}, (acc_bytes) / 4, {.offset_bytes = 0}, {.noc_x = x, .noc_y = y, .addr = daddr});
                        noc.async_write(one, UnicastEndpoint{}, (acc_bytes) / 4, {.offset_bytes = (acc_bytes) / 2}, {.noc_x = x, .noc_y = y, .addr = daddr + (acc_bytes) / 2});
#else
                        noc.async_write(one, UnicastEndpoint{}, acc_bytes, {.offset_bytes = 0}, {.noc_x = x, .noc_y = y, .addr = daddr});
#endif
                    }
                }
            }
            noc.async_writes_flushed();
            for (std::uint32_t y = y0; y <= y1; ++y) {
                for (std::uint32_t x = x0; x <= x1; ++x) {
                    if (x >= gx0 && x < gx1) {
                        continue;
                    }
                    sem_r.up(noc, x, y, 1);
                }
            }
            rout.pop_front(1);
        }
        sem_r.wait_min(1);
        sem_r.set(0);
        rcb.push_back(1);
        }

        std::uint32_t tile = (row0 + r * rstride) * Wt + t0;
        for (std::uint32_t j = 0; j < T; j += blk) {
            const std::uint32_t rem = (j + blk > T) ? (T - j) : blk;
            out0.wait_front(static_cast<uint16_t>(rem));
            std::uint32_t off = 0;
            for (std::uint32_t i = 0; i < rem; ++i) {
                noc.async_write(out0, dst, out_bytes, {.offset_bytes = off}, {.page_id = tile});
                ++tile;
                off += out_bytes;
            }
            noc.async_writes_flushed();
            out0.pop_front(static_cast<uint16_t>(rem));
        }
    }
    noc.async_write_barrier();
#if defined(BH_ACC)
    sm_nb.async_write_barrier();
#endif
    noc.async_atomic_barrier();
}
