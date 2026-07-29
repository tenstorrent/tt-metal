// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// FlashFused reader (two-source): feeds the two-phase flash compute. Reads this core's Q head once
// (one seq-tile chunk, DHt tiles), then the resident prefix K/V (past_k/past_v) chunk-by-chunk into
// cb_k_prefix/cb_v_prefix at the PREFIX tile geometry, then the new/suffix K/V (k/v) into
// cb_k_in/cb_v_in at the SUFFIX tile geometry. The two phases may use different tile heights; each is
// self-consistent (its own tile size + chunk count). When has_past == 0 the prefix phase is empty.
void kernel_main() {
    constexpr uint32_t NQH = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t prefix_Kt = get_compile_time_arg_val(2);
    constexpr uint32_t prefix_Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t prefix_num_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t suffix_Kt = get_compile_time_arg_val(5);
    constexpr uint32_t suffix_Sk_chunk_t = get_compile_time_arg_val(6);
    // PER-CORE under split-KV: only the reducer is given a non-zero suffix chunk count.
    constexpr uint32_t suffix_num_chunks = get_compile_time_arg_val(7);
    constexpr bool has_past = get_compile_time_arg_val(8) == 1;

    constexpr auto q_args = TensorAccessorArgs<9>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pk_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pv_args = TensorAccessorArgs<pk_args.next_compile_time_args_offset()>();
    // PREFIX TILE SKIPPING: when set, the prefix page index indirects through a list of VALID tile
    // indices supplied as runtime args from slot 8. The pi0.5 expert mask is tile-aligned for the case
    // that matters (an absent camera is exactly 8 whole K-tiles, 256 % 32 == 0), so skipping beats
    // masking: no mask tensor is needed at all -- which side-steps the bf8 + dense-mask + 16-row-tile
    // sign inversion -- and it does LESS work than the unmasked path.
    constexpr uint32_t skip_prefix_tiles = get_compile_time_arg_val(pv_args.next_compile_time_args_offset());
    constexpr uint32_t VALID_ARG0 = 8;

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t q_head = get_arg_val<uint32_t>(3);
    const uint32_t kv_head = get_arg_val<uint32_t>(4);
    const uint32_t pk_addr = get_arg_val<uint32_t>(5);
    const uint32_t pv_addr = get_arg_val<uint32_t>(6);
    // Split-KV: this core's first prefix TILE (s * prefix_Kt / kv_splits) and how many suffix chunks
    // it owns (only the reducer takes the suffix). With kv_splits == 1 these are (0, suffix_num_chunks).
    const uint32_t prefix_tile_start = get_arg_val<uint32_t>(7);
    // Logical prefix tile position -> physical tile in past_k/past_v.
    auto prefix_tile = [&](uint32_t pos) -> uint32_t {
        if constexpr (skip_prefix_tiles) {
            return get_arg_val<uint32_t>(VALID_ARG0 + pos);
        } else {
            return pos;
        }
    };
    auto ident_tile = [](uint32_t pos) -> uint32_t { return pos; };

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;      // suffix K
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;      // suffix V
    constexpr uint32_t cb_k_prefix = tt::CBIndex::c_8;  // prefix K
    constexpr uint32_t cb_v_prefix = tt::CBIndex::c_9;  // prefix V

    const uint32_t q_tb = get_tile_size(cb_q_in);
    const uint32_t ks_tb = get_tile_size(cb_k_in);  // suffix K/V tile bytes
    const uint32_t vs_tb = get_tile_size(cb_v_in);
    const uint32_t kp_tb = get_tile_size(cb_k_prefix);  // prefix K/V tile bytes
    const uint32_t vp_tb = get_tile_size(cb_v_prefix);

    const auto q_acc = TensorAccessor(q_args, q_addr, q_tb);
    const auto k_acc = TensorAccessor(k_args, k_addr, ks_tb);
    const auto v_acc = TensorAccessor(v_args, v_addr, vs_tb);
    const auto pk_acc = TensorAccessor(pk_args, pk_addr, kp_tb);
    const auto pv_acc = TensorAccessor(pv_args, pv_addr, vp_tb);

    // Q: head q_head, single seq-tile chunk -> DHt tiles. page = q_head*DHt + d.
    cb_reserve_back(cb_q_in, DHt);
    uint32_t ql1 = get_write_ptr(cb_q_in);
    for (uint32_t d = 0; d < DHt; ++d) {
        noc_async_read_tile(q_head * DHt + d, q_acc, ql1);
        ql1 += q_tb;
    }
    noc_async_read_barrier();
    cb_push_back(cb_q_in, DHt);

    // Read one K/V source chunk-by-chunk. K is written transposed into cb_k: tile (seq=kt, dim=d) ->
    // offset (d*Sk_chunk + kt), i.e. a [DHt x Sk_chunk] tile grid, as the QK^T matmul expects. V stays
    // seq-major [Sk_chunk x DHt]. Page within a source = (kv_head * src_Kt + local) * DHt + d.
#define READ_KV_SOURCE(CB_K, CB_V, K_ACC, V_ACC, K_TB, V_TB, SRC_KT, SK_CHUNK, NUM_CHUNKS, KT_START, MAPPER) \
    for (uint32_t c = 0; c < (NUM_CHUNKS); ++c) {                                                            \
        const uint32_t kt0 = (KT_START) + c * (SK_CHUNK);                                                    \
        cb_reserve_back(CB_K, (SK_CHUNK) * DHt);                                                             \
        const uint32_t k_base = get_write_ptr(CB_K);                                                         \
        cb_reserve_back(CB_V, (SK_CHUNK) * DHt);                                                             \
        uint32_t lv = get_write_ptr(CB_V);                                                                   \
        for (uint32_t kt = 0; kt < (SK_CHUNK); ++kt) {                                                       \
            const uint32_t g = (MAPPER)(kt0 + kt);                                                           \
            const uint32_t base = (kv_head * (SRC_KT) + g) * DHt;                                            \
            for (uint32_t d = 0; d < DHt; ++d) {                                                             \
                noc_async_read_tile(base + d, K_ACC, k_base + (d * (SK_CHUNK) + kt) * (K_TB));               \
                noc_async_read_tile(base + d, V_ACC, lv + d * (V_TB));                                       \
            }                                                                                                \
            lv += DHt * (V_TB);                                                                              \
        }                                                                                                    \
        noc_async_read_barrier();                                                                            \
        cb_push_back(CB_K, (SK_CHUNK) * DHt);                                                                \
        cb_push_back(CB_V, (SK_CHUNK) * DHt);                                                                \
    }

    // Phase 1: resident prefix K/V (own tile geometry). Skipped when there is no past.
    if (has_past) {
        READ_KV_SOURCE(
            cb_k_prefix,
            cb_v_prefix,
            pk_acc,
            pv_acc,
            kp_tb,
            vp_tb,
            prefix_Kt,
            prefix_Sk_chunk_t,
            prefix_num_chunks,
            prefix_tile_start,
            prefix_tile);
    }
    // Phase 2: new suffix K/V (model tiny tile).
    READ_KV_SOURCE(
        cb_k_in, cb_v_in, k_acc, v_acc, ks_tb, vs_tb, suffix_Kt, suffix_Sk_chunk_t, suffix_num_chunks, 0, ident_tile);

    (void)NQH;
}
