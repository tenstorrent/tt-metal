// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_msa packed-group reader (16 query heads per KV group). A "group" is up to G consecutive query
// tokens of one KV group; token slot s of the group owns rows [16*s, 16*s+16) of the group's Q tile rows, so a
// 32-row tile row holds two tokens (slot 2r on top, 2r+1 on the bottom). Each distinct selected block of the
// group (the union of the tokens' top-k rows) is gathered ONCE and streamed to compute, which processes it
// for every tile row that holds a token that selected it; per-token -inf masks hide it from the other token of
// that row and apply the token-level causal mask on each token's own (diagonal) block.
//
// Invariants this kernel establishes for compute:
//  - Union entry 0 (the "lead" block) is selected by every token of the group, so every row's first update is a
//    block it can see (a finite running max: no row starts fully masked). Tokens are added to a group only while
//    such a common block exists; a token that would break it starts the next group.
//  - A token's selection is matched occurrence-by-occurrence to union entries (a block a token lists twice gets
//    two entries), so each token attends exactly the blocks of its own row, with the same multiplicity.
//  - Per token, only its first entry for its diagonal block is causally masked (the legacy kernel's rule).
//
// Per group the reader sends: Q rows (cb_q_rm, 16*G rows; missing slots zero-filled), G half-tile partial-column
// mask tiles (cb_vmask; only slots with a split boundary key-tile are written), one control page (cb_ctrl:
// header + one word per union block), then the union blocks (K/V double-buffered, gathered half by this
// kernel and half by the writer, which receives {block, is_last, k/v slot address, g} in cb_kreq).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"  // per-NoC trid-ring (K_TRID_RING knob)
#include "block_cyclic_remap.hpp"      // tt::block_cyclic::logical_to_physical_page (block-cyclic cache remap)

constexpr uint32_t sentinel = 0xFFFFFFFFu;

// Control page layout (uint32 words), shared with the compute kernel.
//   [0] n_union  [1] g (tokens in the group)
//   [2 + s]       slot s's diagonal-block geometry: boundary key-tile | boundary column << 8 (key-tiles past the
//                 boundary are fully masked; the boundary key-tile gets a partial-column mask when column > 0,
//                 else it is fully masked too; boundary key-tile == Skt -> nothing masked)
//   [2 + G + u]   union block u: bits [0,G) = slots that selected it, bits [8,8+G) = slots for which it is
//                 the (first entry of their) diagonal block.

// Half-tile partial-column mask: rows of `half` (0: rows 0-15 = faces 0,1; 1: rows 16-31 = faces 2,3) get -inf
// at columns >= col; everything else 0. bf16 tile, faces of 16x16, 2 values per word (low = even column).
FORCE_INLINE void fill_half_partial_tile_bf16(uint32_t l1_addr, uint32_t half, uint32_t col) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    constexpr uint32_t words_per_face = 128;
    for (uint32_t f = 0; f < 4; ++f) {
        volatile tt_l1_ptr uint32_t* fp = ptr + f * words_per_face;
        if ((f >> 1) != half) {
            for (uint32_t i = 0; i < words_per_face; ++i) {
                fp[i] = 0;
            }
            continue;
        }
        const uint32_t face_col0 = (f & 1) * 16;
        uint32_t row_words[8];
        for (uint32_t w = 0; w < 8; ++w) {
            const uint32_t c0 = face_col0 + 2 * w;
            const uint32_t lo = (c0 >= col) ? 0xFF80u : 0u;
            const uint32_t hi = (c0 + 1 >= col) ? 0xFF80u : 0u;
            row_words[w] = lo | (hi << 16);
        }
        for (uint32_t r = 0; r < 16; ++r) {
            for (uint32_t w = 0; w < 8; ++w) {
                fp[r * 8 + w] = row_words[w];
            }
        }
    }
}

void kernel_main() {
    constexpr uint32_t H_logical = get_compile_time_arg_val(0);  // 16
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t topk = get_compile_time_arg_val(2);
    constexpr uint32_t n_kv = get_compile_time_arg_val(3);
    constexpr uint32_t q_row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t idx_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(6);
    constexpr uint32_t v_tiles_per_block = get_compile_time_arg_val(7);
    constexpr uint32_t k_half = get_compile_time_arg_val(8);  // writer gathers [0, half)
    constexpr uint32_t v_half = get_compile_time_arg_val(9);
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(10);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(11);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(12);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(13);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(14);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(15);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(16);
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(17);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(19);
    constexpr bool CAUSAL_MASK_ENABLED = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t block_size = get_compile_time_arg_val(21);
    constexpr uint32_t G = get_compile_time_arg_val(22);  // max tokens per group (even, <= 8)
    constexpr bool block_cyclic = get_compile_time_arg_val(23) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(24);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(25);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(26);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(27);
    constexpr auto q_args = TensorAccessorArgs<28, 0>();
    constexpr auto k_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr auto v_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();
    constexpr auto idx_args =
        TensorAccessorArgs<v_args.next_compile_time_args_offset(), v_args.next_common_runtime_args_offset()>();
    static_assert(G >= 2 && G <= 8 && (G % 2) == 0, "packed group size must be 2, 4, 6 or 8");
    static_assert(H_logical == 16, "packed groups put two 16-head tokens in one 32-row tile row");
    constexpr uint32_t rows_per_group = H_logical * G;
    constexpr uint32_t keys_per_tile = 32;  // tile width
    constexpr uint32_t Skt = block_size / keys_per_tile;
    constexpr uint32_t hdr_words = 2 + G;

    // Runtime args: same slots as the legacy reader (SparseSDPAMsaOperation::ReaderArg).
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t idx_addr = get_arg_val<uint32_t>(3);
    const uint32_t work_start = get_arg_val<uint32_t>(4);
    const uint32_t work_count = get_arg_val<uint32_t>(5);
    const uint32_t k_batch_tile_offset = get_arg_val<uint32_t>(6);
    const uint32_t v_batch_tile_offset = get_arg_val<uint32_t>(7);
    uint32_t k_group_tile_stride = 0;
    uint32_t v_group_tile_stride = 0;
    if constexpr (n_kv > 1) {
        k_group_tile_stride = get_arg_val<uint32_t>(8);
        v_group_tile_stride = get_arg_val<uint32_t>(9);
    }
    const uint32_t chunk_start_local = CAUSAL_MASK_ENABLED ? get_arg_val<uint32_t>(10) : 0;
    const uint32_t straddle_row = CAUSAL_MASK_ENABLED ? get_arg_val<uint32_t>(11) : 0;
    const uint32_t straddle_jump = CAUSAL_MASK_ENABLED ? get_arg_val<uint32_t>(12) : 0;

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_in), v_cb(cb_v_in), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    experimental::CB kreq_cb(cb_kreq), kack_cb(cb_kack), vmask_cb(cb_vmask);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    // Reader-internal scratch (reserved once, reused): G block-id rows, then the union's block ids and masks.
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    volatile tt_l1_ptr uint32_t* uni_id = rows + G * topk;
    volatile tt_l1_ptr uint32_t* uni_mask = uni_id + G * topk;

    uint32_t tok = work_start;
    uint32_t kv_group = 0;
    if constexpr (n_kv > 1) {
        kv_group = work_start / S;
        tok = work_start - kv_group * S;
    }
    uint32_t remaining = work_count;
    while (remaining > 0) {
        // ---- candidate tokens: up to G, never across a KV group ----
        uint32_t g_max = remaining < G ? remaining : G;
        if (S - tok < g_max) {
            g_max = S - tok;
        }
        for (uint32_t j = 0; j < g_max; ++j) {
            noc.async_read(
                idx, idx_cb, idx_row_bytes, {.page_id = kv_group * S + tok + j}, {.offset_bytes = j * idx_row_bytes});
        }
        noc.async_read_barrier();

        uint32_t nv[G];
        for (uint32_t j = 0; j < g_max; ++j) {
            // Valid blocks are a contiguous prefix; binary search the first sentinel (legacy rule: >= 1 block).
            volatile tt_l1_ptr uint32_t* r = rows + j * topk;
            uint32_t lo = 0, hi = topk;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (r[mid] == sentinel) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            nv[j] = lo == 0 ? 1 : lo;
        }

        // ---- union of the tokens' selections, keeping a block every token selected (the lead) ----
        // Token 0's ids enter verbatim (in its top-k order); a later token's occurrence of a block takes the
        // first entry of that block it does not own yet, else appends a new entry.
        uint32_t n_union = nv[0];
        for (uint32_t c = 0; c < n_union; ++c) {
            uni_id[c] = rows[c];
            uni_mask[c] = 1u;
        }
        uint32_t g = 1;
        for (uint32_t j = 1; j < g_max; ++j) {
            const uint32_t bit = 1u << j;
            const uint32_t n_before = n_union;
            volatile tt_l1_ptr uint32_t* r = rows + j * topk;
            for (uint32_t c = 0; c < nv[j]; ++c) {
                const uint32_t b = r[c];
                uint32_t e = 0;
                while (e < n_union && !(uni_id[e] == b && (uni_mask[e] & bit) == 0)) {
                    ++e;
                }
                if (e == n_union) {
                    uni_id[n_union] = b;
                    uni_mask[n_union] = 0;
                    ++n_union;
                }
                uni_mask[e] |= bit;
            }
            const uint32_t all = (bit << 1) - 1;
            bool has_lead = false;
            for (uint32_t e = 0; e < n_before && !has_lead; ++e) {
                has_lead = (uni_mask[e] & all) == all;
            }
            if (!has_lead) {
                // Roll token j back; it starts the next group.
                n_union = n_before;
                for (uint32_t e = 0; e < n_union; ++e) {
                    uni_mask[e] &= ~bit;
                }
                break;
            }
            g = j + 1;
        }
        // Move the first common entry to the front (identity when token 0 lists it first, the common case).
        {
            const uint32_t all = (1u << g) - 1;
            uint32_t lead = 0;
            while ((uni_mask[lead] & all) != all) {
                ++lead;
            }
            if (lead != 0) {
                const uint32_t lid = uni_id[lead], lmask = uni_mask[lead];
                for (uint32_t e = lead; e > 0; --e) {
                    uni_id[e] = uni_id[e - 1];
                    uni_mask[e] = uni_mask[e - 1];
                }
                uni_id[0] = lid;
                uni_mask[0] = lmask;
            }
        }

        // ---- per-slot causal geometry; flag each token's first diagonal entry ----
        uint32_t bt[G], bcol[G];
        for (uint32_t s = 0; s < G; ++s) {
            bt[s] = Skt;
            bcol[s] = 0;
        }
        uint32_t diag_bits[G];  // union entry index of slot s's diagonal block, or sentinel
        for (uint32_t s = 0; s < G; ++s) {
            diag_bits[s] = sentinel;
        }
        if constexpr (CAUSAL_MASK_ENABLED) {
            for (uint32_t s = 0; s < g; ++s) {
                const uint32_t t = tok + s;
                const uint32_t p = chunk_start_local + t + (t >= straddle_row ? straddle_jump : 0);
                const uint32_t diag_block = p / block_size;
                const uint32_t first_masked = (p % block_size) + 1;
                bt[s] = first_masked / keys_per_tile;
                bcol[s] = first_masked % keys_per_tile;
                // The legacy kernel masks the first occurrence in the token's own order. Token 0's entries are in
                // its order; a later token's occurrences map to that block's entries in increasing index.
                for (uint32_t e = 0; e < n_union; ++e) {
                    if (uni_id[e] == diag_block && (uni_mask[e] & (1u << s))) {
                        diag_bits[s] = e;
                        break;
                    }
                }
            }
        }

        // ---- partial-column mask tiles (one page per slot; only slots with a split boundary are written) ----
        vmask_cb.reserve_back(G);
        {
            constexpr uint32_t mask_tile_bytes = get_tile_size(cb_vmask);
            const uint32_t base = vmask_cb.get_write_ptr();
            for (uint32_t s = 0; s < g; ++s) {
                if (diag_bits[s] != sentinel && bcol[s] > 0) {
                    fill_half_partial_tile_bf16(base + s * mask_tile_bytes, s & 1, bcol[s]);
                }
            }
        }
        vmask_cb.push_back(G);

        // ---- Q rows of the group (missing slots zero-filled) ----
        q_cb.reserve_back(rows_per_group);
        for (uint32_t s = 0; s < g; ++s) {
            for (uint32_t h = 0; h < H_logical; ++h) {
                const uint32_t q_head = h + kv_group * H_logical;
                noc.async_read(
                    q,
                    q_cb,
                    q_row_bytes,
                    {.page_id = q_head * S + tok + s},
                    {.offset_bytes = (s * H_logical + h) * q_row_bytes});
            }
        }
        noc.async_read_barrier();
        if (g < G) {
            for (uint32_t s = g; s < G; ++s) {
                noc.async_write_zeros(q_cb, H_logical * q_row_bytes, {.offset_bytes = s * H_logical * q_row_bytes});
            }
            noc.write_zeros_l1_barrier();
        }
        q_cb.push_back(rows_per_group);

        // ---- control page ----
        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[0] = n_union;
            cp[1] = g;
            for (uint32_t s = 0; s < G; ++s) {
                cp[2 + s] = (s < g && diag_bits[s] != sentinel) ? (bt[s] | (bcol[s] << 8)) : Skt;
            }
            for (uint32_t e = 0; e < n_union; ++e) {
                uint32_t dmask = 0;
                for (uint32_t s = 0; s < g; ++s) {
                    if (diag_bits[s] == e) {
                        dmask |= 1u << s;
                    }
                }
                cp[hdr_words + e] = uni_mask[e] | (dmask << 8);
            }
        }
        ctrl_cb.push_back(1);

        // ---- stream the union blocks (double-buffered K/V; the writer gathers the lower halves) ----
        for (uint32_t e = 0; e < n_union; ++e) {
            const uint32_t block_id = uni_id[e];
            ASSERT(block_id != sentinel);
            const uint32_t phys_block = tt::block_cyclic::
                logical_to_physical_page<block_cyclic, bc_chunk_local, bc_sp, bc_shard_stride_gap, bc_slab_stride_gap>(
                    block_id);
            uint32_t k_tile0 = k_batch_tile_offset + phys_block * k_tiles_per_block;
            uint32_t v_tile0 = v_batch_tile_offset + phys_block * v_tiles_per_block;
            if constexpr (n_kv > 1) {
                k_tile0 += kv_group * k_group_tile_stride;
                v_tile0 += kv_group * v_group_tile_stride;
            }
            k_cb.reserve_back(k_tiles_per_block);
            v_cb.reserve_back(v_tiles_per_block);
            const uint32_t k_slot = k_cb.get_write_ptr();
            const uint32_t v_slot = v_cb.get_write_ptr();

            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = phys_block;
                rq[1] = (e == n_union - 1);  // writer drains the group's output after its last block
                rq[2] = k_slot;
                rq[3] = v_slot;
                rq[4] = g;
            }
            kreq_cb.push_back(1);

            sparse_sdpa_msa::TridRing ring{noc};  // K/V upper halves share one ring.
            for (uint32_t i = k_half; i < k_tiles_per_block; ++i) {
                ring.read_to(k, k_slot + i * k_tile_bytes, k_tile_bytes, k_tile0 + i);
            }
            for (uint32_t i = v_half; i < v_tiles_per_block; ++i) {
                ring.read_to(v, v_slot + i * v_tile_bytes, v_tile_bytes, v_tile0 + i);
            }
            ring.drain();
            kack_cb.wait_front(1);  // writer's lower halves landed in the same slot
            kack_cb.pop_front(1);
            k_cb.push_back(k_tiles_per_block);
            v_cb.push_back(v_tiles_per_block);
        }

        tok += g;
        remaining -= g;
        if constexpr (n_kv > 1) {
            if (tok == S) {
                tok = 0;
                ++kv_group;
            }
        }
    }
}
