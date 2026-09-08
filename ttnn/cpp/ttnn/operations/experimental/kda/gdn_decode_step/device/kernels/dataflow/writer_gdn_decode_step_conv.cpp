// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for the fused-conv variant (packed layout). The packed history is private per value head (no cross-core
// hazard); the shift (slot0 <- slot1, ..., slot3 <- new token) is snapshotted at kernel start and written after the
// state, i.e. after this core's reader has consumed the old history.
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
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
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
    for (uint32_t i = 0; i < wi_count; ++i) {
        const uint32_t h = wi_start + i;
        const uint32_t hk = h / rf;
        // wshift = [slot1, slot2, slot3, new token]  ->  slots 0..3
        wshift.reserve_back(4);
        zero_reserved(wshift, noc, 4);
        read_tiles_at(hist_acc, wshift, noc, h * 4 + 1, 3, 0);
        pack_head_tile<Kt, Vt, Nk>(qkv_acc, wshift, noc, hk, h, 3);
        noc.async_read_barrier();
        wshift.push_back(4);
        // The state write below waits for the compute output, which in turn consumed the reader's copy of the history:
        // only after that is it safe to overwrite slots 1..3 (the reader and writer share this core's history pages).
        write_tiles(state_acc, hnew, noc, h * KV, KV);  // in-place state update
        write_tiles(hist_acc, wshift, noc, h * 4, 4);   // slot0 <- slot1, ..., slot3 <- new token (8 KB)
        write_tiles(out_acc, out, noc, h * Vt, Vt);
    }
}
