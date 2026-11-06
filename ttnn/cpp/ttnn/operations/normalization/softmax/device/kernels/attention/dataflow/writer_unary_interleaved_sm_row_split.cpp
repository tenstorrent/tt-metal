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
    Semaphore sem_rmax(sem::rmax_sem);
    Semaphore sem_rsum(sem::rsum_sem);
    Semaphore sem_m(sem::m_sem);
    Semaphore sem_r(sem::r_sem);
    const auto dst = TensorAccessor(tensor::dst);
    const std::uint32_t out_bytes = out0.get_entry_size();
    const std::uint32_t im_bytes = pmax.get_entry_size();
    const std::uint32_t acc_bytes = psum.get_entry_size();

    for (std::uint32_t r = 0; r < num_rows; ++r) {
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
            sem_rmax.up(noc, comb_x, comb_y, 1);
            noc.async_atomic_barrier();
        }
        pmax.pop_front(1);

        mcb.reserve_back(1);
        if (is_comb) {
            sem_rmax.wait(active);
            sem_rmax.set(0);
            rmax.push_back(static_cast<uint16_t>(active));
            mout.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> one(mout.get_read_ptr());
                const std::uint32_t daddr = mcb.get_write_ptr();
                for (std::uint32_t y = y0; y <= y1; ++y) {
                    for (std::uint32_t x = x0; x <= x1; ++x) {
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
                    sem_m.up(noc, x, y, 1);
                }
            }
            noc.async_atomic_barrier();
            mout.pop_front(1);
        }
        sem_m.wait_min(1);
        sem_m.set(0);
        mcb.push_back(1);

        if (is_comb) {
            rsum.reserve_back(static_cast<uint16_t>(nchunk));
        }
        for (std::uint32_t k = 0; k < nck; ++k) {
            psum.wait_front(1);
            CoreLocalMem<std::uint32_t> src(psum.get_read_ptr());
            noc.async_write(
                src,
                UnicastEndpoint{},
                acc_bytes,
                {.offset_bytes = 0},
                {.noc_x = comb_x, .noc_y = comb_y, .addr = rsum.get_write_ptr() + (ck0 + k) * acc_bytes});
            noc.async_write_barrier();
            psum.pop_front(1);
        }
        sem_rsum.up(noc, comb_x, comb_y, 1);
        noc.async_atomic_barrier();

        rcb.reserve_back(1);
        if (is_comb) {
            sem_rsum.wait(active);
            sem_rsum.set(0);
            rsum.push_back(static_cast<uint16_t>(nchunk));
            rout.wait_front(1);
            {
                CoreLocalMem<std::uint32_t> one(rout.get_read_ptr());
                const std::uint32_t daddr = rcb.get_write_ptr();
                for (std::uint32_t y = y0; y <= y1; ++y) {
                    for (std::uint32_t x = x0; x <= x1; ++x) {
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
                    sem_r.up(noc, x, y, 1);
                }
            }
            noc.async_atomic_barrier();
            rout.pop_front(1);
        }
        sem_r.wait_min(1);
        sem_r.set(0);
        rcb.push_back(1);

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
            noc.async_write_barrier();
            out0.pop_front(static_cast<uint16_t>(rem));
        }
    }
}
