// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for the batched fused-conv variant. Per user: snapshot the packed history slots 1..3 + the new token, then
// after the compute's state output (which implies this core's reader consumed the old history) write the state in
// place and the shifted history. After the last user: write the group's rows of the head's output tiles.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"

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

template <uint32_t Kt, uint32_t Vt, uint32_t Nk, uint32_t Nv>
TT_KERNEL void writer(uint32_t head, uint32_t u0, uint32_t nu) {
    const auto state_acc = TensorAccessor(tensor::state_out);
    const auto out_acc = TensorAccessor(tensor::out);
    const auto qkv_acc = TensorAccessor(tensor::qkv_w);
    const auto hist_acc = TensorAccessor(tensor::hist_w);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    DataflowBuffer wshift(dfb::wshift);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t rf = Nv / Nk;
    const uint32_t h = head;
    const uint32_t hk = h / rf;
    for (uint32_t ui = 0; ui < nu; ++ui) {
        const uint32_t b = u0 + ui;
        const uint32_t bh = b * Nv + h;
        wshift.reserve_back(4);
        zero_reserved(wshift, noc, 4);
        read_tiles_at(hist_acc, wshift, noc, bh * 4 + 1, 3, 0);
        pack_head_tile_user<Kt, Vt, Nk>(qkv_acc, wshift, noc, hk, h, b, 3);
        noc.async_read_barrier();
        wshift.push_back(4);
        write_tiles(state_acc, hnew, noc, bh * KV, KV);  // waits for compute -> old history already consumed
        write_tiles(hist_acc, wshift, noc, bh * 4, 4);   // slot0 <- slot1, ..., slot3 <- new token
    }
    write_rows(out_acc, out, noc, h * Vt, Vt, u0, nu);  // the group's rows of this head's output tiles
}
