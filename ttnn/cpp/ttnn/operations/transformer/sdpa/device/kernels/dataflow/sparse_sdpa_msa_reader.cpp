// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_msa reader: for each query token, read Q rows and selected pre-tiled K/V blocks. Reader and writer
// split each block gather into upper/lower tile halves. Masking is represented only by sentinel block ids.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"  // per-NoC trid-ring (K_TRID_RING knob)
#include "dataflow_common.hpp"         // fill_vertical_tile_bf16 (causal partial-column mask tile)

constexpr uint32_t sentinel = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t H_logical = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t topk = get_compile_time_arg_val(3);
    constexpr uint32_t n_kv = get_compile_time_arg_val(4);
    constexpr uint32_t q_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t idx_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(7);
    constexpr uint32_t v_tiles_per_block = get_compile_time_arg_val(8);
    constexpr uint32_t k_half = get_compile_time_arg_val(9);  // writer gathers [0, half)
    constexpr uint32_t v_half = get_compile_time_arg_val(10);

    // CB ids match the factory's reader compile-arg block.
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(11);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(12);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(13);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(14);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(15);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(16);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(17);

    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(18);  // K is tiled: per-tile read size
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(19);  // V is tiled: per-tile read size

    // Causal masking (token-level diagonal-block mask)
    constexpr bool CAUSAL_MASK_ENABLED = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t block_size = get_compile_time_arg_val(21);  // tokens per block (for p%bs, p/bs)
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(22);    // per-token partial-column mask tile

    // K/V use RuntimeTensorShape so T can vary without recompilation.
    constexpr auto q_args = TensorAccessorArgs<23, 0>();
    constexpr auto k_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr auto v_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();
    constexpr auto idx_args =
        TensorAccessorArgs<v_args.next_compile_time_args_offset(), v_args.next_common_runtime_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t idx_addr = get_arg_val<uint32_t>(3);
    const uint32_t work_start = get_arg_val<uint32_t>(4);
    const uint32_t work_count = get_arg_val<uint32_t>(5);
    // Indexed-cache slot offsets; zero when cache_batch_idx is unset.
    const uint32_t k_batch_tile_offset = get_arg_val<uint32_t>(6);
    const uint32_t v_batch_tile_offset = get_arg_val<uint32_t>(7);
    uint32_t k_group_tile_stride = 0;
    uint32_t v_group_tile_stride = 0;
    if constexpr (n_kv > 1) {
        k_group_tile_stride = get_arg_val<uint32_t>(8);
        v_group_tile_stride = get_arg_val<uint32_t>(9);
    }
    // Per-device global position of this core's query row 0 (chunk_start_idx + rank*S); patched at dispatch.
    const uint32_t chunk_start_local = CAUSAL_MASK_ENABLED ? get_arg_val<uint32_t>(10) : 0;
    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_in), v_cb(cb_v_in), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    experimental::CB kreq_cb(cb_kreq), kack_cb(cb_kack);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    // Reader-internal scratch for one token's block-id row (reserved once, reused).
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    uint32_t tok = work_start;
    uint32_t kv_group = 0;
    if constexpr (n_kv > 1) {
        kv_group = work_start / S;
        tok = work_start - kv_group * S;
    }
    for (uint32_t work = 0; work < work_count; ++work) {
        // Q: logical head rows plus zero-filled padded heads so compute always sees full 32-head tiles.
        q_cb.reserve_back(H);
        for (uint32_t h = 0; h < H_logical; ++h) {
            uint32_t q_head = h;
            if constexpr (n_kv > 1) {
                q_head += kv_group * H_logical;
            }
            noc.async_read(q, q_cb, q_row_bytes, {.page_id = q_head * S + tok}, {.offset_bytes = h * q_row_bytes});
        }
        noc.async_read_barrier();
        if constexpr (H_logical < H) {
            noc.async_write_zeros(q_cb, (H - H_logical) * q_row_bytes, {.offset_bytes = H_logical * q_row_bytes});
            noc.write_zeros_l1_barrier();
        }
        q_cb.push_back(H);

        // Block-id row for this token.
        uint32_t idx_page = tok;
        if constexpr (n_kv > 1) {
            idx_page += kv_group * S;
        }
        noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = idx_page}, {.offset_bytes = 0});
        noc.async_read_barrier();

        // Binary search the first sentinel; valid blocks are a contiguous prefix.
        uint32_t nv_blocks = topk;
        {
            uint32_t lo = 0, hi = topk;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (idx_ptr[mid] == sentinel) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            nv_blocks = lo == 0 ? 1 : lo;  // ASSERT below traps all-sentinel rows.
        }
        const uint32_t n_active = nv_blocks;  // each active chunk is one full block

        // Causal control: locate the diagonal block (the query's own) among the selected blocks and the
        // within-block boundary, so compute masks the future tokens inside it. Non-causal -> sentinel, no mask.
        uint32_t diag_chunk = sentinel;
        uint32_t boundary_tile = 0;  // first fully-masked key-tile within the diagonal block
        uint32_t boundary_col = 0;   // within-tile column where masking starts (0 -> boundary_tile fully masked)
        if constexpr (CAUSAL_MASK_ENABLED) {
            const uint32_t p = chunk_start_local + tok;  // global query position
            const uint32_t diag_block = p / block_size;
            for (uint32_t c = 0; c < n_active; ++c) {  // block_ids are topk-ordered (unsorted) -> linear scan
                if (idx_ptr[c] == diag_block) {
                    diag_chunk = c;
                    break;
                }
            }
            const uint32_t first_masked = (p % block_size) + 1;  // local offset of the first future token
            boundary_tile = first_masked / keys_per_tile;
            boundary_col = first_masked % keys_per_tile;
        }
        [[maybe_unused]] const bool build_vmask = (diag_chunk != sentinel) && (boundary_col > 0);

        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[0] = n_active;
            cp[1] = diag_chunk;     // chunk index of the diagonal block (sentinel = none -> no token mask)
            cp[2] = boundary_tile;  // key-tiles >= this are fully masked in the diagonal block
            cp[3] = boundary_col;   // partial-column boundary within boundary_tile (0 -> none)
        }
        ctrl_cb.push_back(1);

        // Build the partial-column mask tile for the boundary key-tile (only when it splits a tile).
        if constexpr (CAUSAL_MASK_ENABLED) {
            if (build_vmask) {
                constexpr uint32_t mask_tile_bytes = get_tile_size(cb_vmask);
                experimental::CB(cb_vmask).reserve_back(1);
                fill_vertical_tile_bf16<mask_tile_bytes>(noc, cb_vmask, 0, boundary_col);
                experimental::CB(cb_vmask).push_back(1);
            }
        }

        for (uint32_t chunk = 0; chunk < n_active; ++chunk) {
            const uint32_t block_id = idx_ptr[chunk];
            // Producer contract requires at least one valid block and no sentinels in the active prefix.
            ASSERT(block_id != sentinel);

            // Reader reserves the whole block; writer fills the lower half, reader fills the upper half.
            uint32_t k_tile0 = k_batch_tile_offset + block_id * k_tiles_per_block;
            uint32_t v_tile0 = v_batch_tile_offset + block_id * v_tiles_per_block;
            if constexpr (n_kv > 1) {
                k_tile0 += kv_group * k_group_tile_stride;
                v_tile0 += kv_group * v_group_tile_stride;
            }
            k_cb.reserve_back(k_tiles_per_block);
            v_cb.reserve_back(v_tiles_per_block);

            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = block_id;
                rq[1] = (chunk == n_active - 1);  // writer ends its gather phase for the token on the last block
            }
            kreq_cb.push_back(1);

            sparse_sdpa_msa::TridRing ring{noc};  // K/V upper halves share one ring.
            for (uint32_t i = k_half; i < k_tiles_per_block; ++i) {
                ring.read(k, k_cb, k_tile_bytes, k_tile0 + i, i * k_tile_bytes);
            }
            for (uint32_t i = v_half; i < v_tiles_per_block; ++i) {
                ring.read(v, v_cb, v_tile_bytes, v_tile0 + i, i * v_tile_bytes);
            }
            ring.drain();           // this NoC's upper halves landed
            kack_cb.wait_front(1);  // writer's lower halves landed in the same L1
            kack_cb.pop_front(1);
            k_cb.push_back(k_tiles_per_block);
            v_cb.push_back(v_tiles_per_block);
        }

        ++tok;
        if constexpr (n_kv > 1) {
            if (tok == S) {
                tok = 0;
                ++kv_group;
            }
        }
    }
}
