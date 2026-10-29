// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for the batched fused-conv variant with value blocks (VB > 1; writer_gdn_decode_step_conv.cpp stays the
// VB = 1 program). One core = value head `head`, users [u0, u0 + nu), state / output column tiles [v0, v0 + Vs),
// v0 = s*Vs. WRITE-ONLY on this kernel's NOC_1 (the history snapshot is read by the reader on NOC_0 into `wshift`):
//  - per user: this block's KVs state tiles (hnew entry i*Vs + j -> tile (i, v0 + j) of the [Kt, Vt] block) in place;
//  - after the last user: the exchange of the users' pre-norm output tiles within the head: acc_mine (nu*Vs fp32
//    tiles, exact bytes; user ui at ui*Vs) -> acc_x[ui*Vt + v0 ..] on every block (itself included) by unicast NoC
//    writes. Handshake as writer_gdn_spec_step.cpp (both semaphores start at 0 on every launch; no in-kernel reset):
//    each block raises `ready` on every block once it runs (its acc_x is reserved), waits for VB before writing,
//    write-barriers, then raises `valid` on every block and waits for VB before handing acc_x to the compute. No
//    multicast; 2 * VB atomics and VB * nu writes per core;
//  - block 0 only: the shifted history of every user, AFTER the exchange: `valid` = VB means every block's compute
//    produced all its outputs, so every block's reader has consumed the old history (it is shifted in place);
//  - the group's rows of this block's Vs output tiles.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

using namespace gdn_step_df;

namespace {
template <typename Accessor>
inline void write_tiles(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count) {
    dfb.wait_front(count);
    const uint32_t entry = dfb.get_entry_size();
    for (uint32_t t = 0; t < count; ++t) {
        noc.async_write(dfb, acc, entry, {.offset_bytes = t * entry}, {.page_id = first_page + t});
    }
    noc.async_write_barrier();
    dfb.pop_front(count);
}
}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t Nv, uint32_t VB>
TT_KERNEL void writer(
    uint32_t head,
    uint32_t u0,
    uint32_t nu,
    uint32_t s,
    uint32_t peer_x0,
    uint32_t peer_y0,
    uint32_t peer_x1,
    uint32_t peer_y1,
    uint32_t peer_x2,
    uint32_t peer_y2,
    uint32_t peer_x3,
    uint32_t peer_y3) {
    const auto state_acc = TensorAccessor(tensor::state_out);
    const auto out_acc = TensorAccessor(tensor::out);
    const auto hist_acc = TensorAccessor(tensor::hist_w);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    DataflowBuffer wshift(dfb::wshift);
    DataflowBuffer acc_mine(dfb::acc_mine);
    DataflowBuffer acc_x(dfb::acc_x);
    Semaphore ready(sem::ready);
    Semaphore valid(sem::valid);
    Noc noc;
    static_assert(VB > 1 && VB <= 4 && Vt % VB == 0, "value blocks must divide Vt (2..4 peers)");
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Vs = Vt / VB;
    constexpr uint32_t KVs = Kt * Vs;
    const uint32_t v0 = s * Vs;
    const uint32_t h = head;
    const uint32_t peer_x[4] = {peer_x0, peer_x1, peer_x2, peer_x3};
    const uint32_t peer_y[4] = {peer_y0, peer_y1, peer_y2, peer_y3};

    // acc_x is this program's buffer from launch on: reserve it now (same L1 address on every block: same spec, first
    // use) and tell every block of the head it may write into it
    acc_x.reserve_back(nu * Vt);
    const uint32_t acc_x_base = acc_x.get_write_ptr();
    for (uint32_t p = 0; p < VB; ++p) {
        ready.up(noc, peer_x[p], peer_y[p], 1);
    }

    // ---- per user: this block's state tiles in place
    {
        const uint32_t entry = hnew.get_entry_size();
        for (uint32_t ui = 0; ui < nu; ++ui) {
            const uint32_t bh = (u0 + ui) * Nv + h;
            hnew.wait_front(KVs);
            const uint32_t page0 = bh * KV + v0;
            for (uint32_t i = 0; i < Kt; ++i) {
                for (uint32_t j = 0; j < Vs; ++j) {
                    noc.async_write(
                        hnew,
                        state_acc,
                        entry,
                        {.offset_bytes = (i * Vs + j) * entry},
                        {.page_id = page0 + i * Vt + j});
                }
            }
            noc.async_write_barrier();
            hnew.pop_front(KVs);
        }
    }

    // ---- swap the users' pre-norm output tiles within the head: acc_mine[ui*Vs ..] -> acc_x[ui*Vt + v0 ..] everywhere
    {
        acc_mine.wait_front(nu * Vs);
        const uint32_t entry = acc_mine.get_entry_size();
        ready.wait(VB);  // every block of the head runs this program (its acc_x is ours to write)
        for (uint32_t p = 0; p < VB; ++p) {
            for (uint32_t ui = 0; ui < nu; ++ui) {
                noc.async_write(
                    acc_mine,
                    UnicastEndpoint{},
                    Vs * entry,
                    {.offset_bytes = ui * Vs * entry},
                    {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = acc_x_base + (ui * Vt + v0) * entry});
            }
        }
        noc.async_write_barrier();  // the tiles landed before `valid` is raised
        acc_mine.pop_front(nu * Vs);
        for (uint32_t p = 0; p < VB; ++p) {
            valid.up(noc, peer_x[p], peer_y[p], 1);
        }
        valid.wait(VB);  // every block's tiles are in this block's acc_x
        acc_x.push_back(nu * Vt);
    }

    // ---- block 0: the shifted history of every user (slot0 <- slot1, ..., slot3 <- new token)
    if (s == 0) {
        for (uint32_t ui = 0; ui < nu; ++ui) {
            const uint32_t bh = (u0 + ui) * Nv + h;
            write_tiles(hist_acc, wshift, noc, bh * 4, 4);
        }
    }

    write_rows(out_acc, out, noc, h * Vt + v0, Vs, u0, nu);  // the group's rows of this block's output columns
    noc.async_atomic_barrier();                              // retire the ready / valid increments before exit
}
