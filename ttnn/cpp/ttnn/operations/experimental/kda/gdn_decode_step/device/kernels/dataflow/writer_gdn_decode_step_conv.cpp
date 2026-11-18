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

// write_tiles into the two-pool page space (see read_tiles_at_2)
template <bool two_pools, typename Acc>
inline void write_tiles_2(
    const Acc& a, const Acc& b, uint32_t pages_a, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t count) {
    if constexpr (two_pools) {
        const bool in_b = page >= pages_a;
        write_tiles(in_b ? b : a, dfb, noc, in_b ? page - pages_a : page, count);
    } else {
        write_tiles(a, dfb, noc, page, count);
    }
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
#ifdef GDN_STATE_PAGE_IDX
    const auto pidx_acc = TensorAccessor(tensor::pidx);
    DataflowBuffer pidx(dfb::pidx_w);
    read_state_page_idx(pidx_acc, pidx, noc, h, Nv);
    noc.async_read_barrier();
#endif
#ifdef GDN_STATE_POOL2
    const auto state2_acc = same_layout_accessor(state_acc, tensor::state2_out);
    constexpr bool state2 = true;
    constexpr uint32_t state_pages_a = GDN_STATE_POOL2;
#else
    const auto& state2_acc = state_acc;
    constexpr bool state2 = false;
    constexpr uint32_t state_pages_a = 0;
#endif
#ifdef GDN_HIST_POOL2
    const auto hist2_acc = same_layout_accessor(hist_acc, tensor::hist2_w);
    constexpr bool hist2 = true;
    constexpr uint32_t hist_pages_a = GDN_HIST_POOL2;
#else
    const auto& hist2_acc = hist_acc;
    constexpr bool hist2 = false;
    constexpr uint32_t hist_pages_a = 0;
#endif
    const uint32_t hk = h / rf;
    for (uint32_t ui = 0; ui < nu; ++ui) {
        const uint32_t b = u0 + ui;
#ifdef GDN_STATE_PAGE_IDX
        const uint32_t state_page = state_page_idx_at(pidx, 0, b);
        const uint32_t hist_page = state_page_idx_at(pidx, 1, b);
#else
        const uint32_t state_page = (b * Nv + h) * KV;
        const uint32_t hist_page = (b * Nv + h) * 4;
#endif
        wshift.reserve_back(4);
        zero_reserved(wshift, noc, 4);
        read_tiles_at_2<hist2>(hist_acc, hist2_acc, hist_pages_a, wshift, noc, hist_page + 1, 3, 0);
        pack_head_tile_user<Kt, Vt, Nk>(qkv_acc, wshift, noc, hk, h, b, 3);
        noc.async_read_barrier();
#ifdef GDN_STATE_PAGE_IDX
        // the new slot gets both parities, as slots 0..2 already have them (the writer waits on the compute next, so
        // this copy is off the critical path)
        packed_rows_mirror_parity(wshift, 3, 2 * Kt + Vt, b);
#endif
        wshift.push_back(4);
        // waits for compute -> old history already consumed
        write_tiles_2<state2>(state_acc, state2_acc, state_pages_a, hnew, noc, state_page, KV);
        write_tiles_2<hist2>(hist_acc, hist2_acc, hist_pages_a, wshift, noc, hist_page, 4);  // slot s <- s+1, 3 <- new
    }
    write_rows(out_acc, out, noc, h * Vt, Vt, u0, nu);  // the group's rows of this head's output tiles
}
