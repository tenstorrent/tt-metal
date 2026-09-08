// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for the fused-conv variant: stores the new state and output, then shifts the conv history in place
// (cs0 <- cs1, cs1 <- cs2, cs2 <- cs3, cs3 <- new qkv). The history is read at kernel start (before any core can
// have started writing) and written after the compute finishes; q/k tiles shared by rf value heads are written by
// the first head of each key group only.
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
    const auto cs0_acc = TensorAccessor(tensor::cs0_out);
    const auto cs1_acc = TensorAccessor(tensor::cs1_w);
    const auto cs2_acc = TensorAccessor(tensor::cs2_w);
    const auto cs3_acc = TensorAccessor(tensor::cs3_w);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    DataflowBuffer wh1(dfb::wh1);
    DataflowBuffer wh2(dfb::wh2);
    DataflowBuffer wh3(dfb::wh3);
    DataflowBuffer wcur(dfb::wcur);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t rf = Nv / Nk;
    for (uint32_t i = 0; i < wi_count; ++i) {
        const uint32_t h = wi_start + i;
        const uint32_t hk = h / rf;
        // snapshot the history rows before anyone shifts them
        read_head_row<Kt, Vt, Nk>(cs1_acc, wh1, noc, hk, h);
        read_head_row<Kt, Vt, Nk>(cs2_acc, wh2, noc, hk, h);
        read_head_row<Kt, Vt, Nk>(cs3_acc, wh3, noc, hk, h);
        read_head_row<Kt, Vt, Nk>(qkv_acc, wcur, noc, hk, h);
        write_tiles(state_acc, hnew, noc, h * KV, KV);  // in-place state update
        write_tiles(out_acc, out, noc, h * Vt, Vt);
        const bool write_qk = (h % rf) == 0;
        write_head_row<Kt, Vt, Nk>(cs0_acc, wh1, noc, hk, h, write_qk);
        write_head_row<Kt, Vt, Nk>(cs1_acc, wh2, noc, hk, h, write_qk);
        write_head_row<Kt, Vt, Nk>(cs2_acc, wh3, noc, hk, h, write_qk);
        write_head_row<Kt, Vt, Nk>(cs3_acc, wcur, noc, hk, h, write_qk);
    }
}
