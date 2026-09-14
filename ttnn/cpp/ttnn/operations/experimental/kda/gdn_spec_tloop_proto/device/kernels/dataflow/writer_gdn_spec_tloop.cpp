// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// M0a scratch writer: per token, stream the post-token state (16 fp32 tiles = 64 KiB) from the double-buffered
// `hnew` to ring block (t*BH + bh) (token-major, as writer_fused_recurrent_gated_delta_rule.cpp); after the loop
// write rows u*T .. u*T+T-1 of the head's Vt output tiles.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"

using namespace gdn_step_df;

namespace {
// Rows [r0, r0 + nr) of `count` L1 tiles to the same rows of the destination tiles, ONE face-row (32 B for bf16) per
// NoC write: source and destination offsets are equal, so both sit in the same 32 B alignment class whatever the row
// parity (the M2 spec 2.7 probe for the odd-row T = 1 seed write).
template <typename Accessor>
inline void write_rows_single(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, uint32_t r0, uint32_t nr) {
    dfb.wait_front(count);
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t esz = entry / 1024;
    const uint32_t seg = entry / 64;
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t base = t * entry;
        for (uint32_t r = r0; r < r0 + nr; ++r) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = tile_elem_index(r, half * 16) * esz;
                noc.async_write(
                    dfb, acc, seg, {.offset_bytes = base + off}, {.page_id = first_page + t, .offset_bytes = off});
            }
        }
    }
    noc.async_write_barrier();
    dfb.pop_front(count);
}
}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nv,
    uint32_t T,
    uint32_t B,
    uint32_t write_ring,
    uint32_t out_w_tiles,
    uint32_t opt_flags>
TT_KERNEL void writer(uint32_t u, uint32_t h) {
    const auto ring_acc = TensorAccessor(tensor::ring_out);
    const auto out_acc = TensorAccessor(tensor::out);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t BH = B * Nv;
    const uint32_t bh = u * Nv + h;
    const uint32_t tr = (u * T) / 32;  // output tile row of the user's rows
    const uint32_t r0 = (u * T) % 32;
    const uint32_t entry = hnew.get_entry_size();
    for (uint32_t t = 0; t < T; ++t) {
        hnew.wait_front(KV);
        if constexpr (write_ring != 0) {
            const uint32_t page0 = (t * BH + bh) * KV;
            for (uint32_t i = 0; i < KV; ++i) {
                noc.async_write(hnew, ring_acc, entry, {.offset_bytes = i * entry}, {.page_id = page0 + i});
            }
            noc.async_write_barrier();
        }
        hnew.pop_front(KV);
    }
    if constexpr ((opt_flags & 8u) != 0) {
        write_rows_single(out_acc, out, noc, tr * out_w_tiles + h * Vt, Vt, r0, T);
    } else {
        write_rows(out_acc, out, noc, tr * out_w_tiles + h * Vt, Vt, r0, T);
    }
}
