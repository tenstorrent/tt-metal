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
#include "block_cyclic_remap.hpp"      // tt::block_cyclic::logical_to_physical_page (block-cyclic cache remap)
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_msa_common.hpp"  // kreq + ctrl records

namespace kreq = sparse_sdpa_msa::kreq;
namespace ctrl = sparse_sdpa_msa::ctrl;

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

    // CB ids match the factory's reader compile-arg block (meanings: SparseSDPAMsaOperation::Cb).
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

    // Block-cyclic ("slab") cache remap in BLOCK units: baked compile-time so a natural-order cache folds to
    // identity (block_cyclic false). Kept in lockstep with the writer's block remap.
    constexpr bool block_cyclic = get_compile_time_arg_val(23) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(24);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(25);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(26);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(27);

    // Per-core K/V block cache: KV_CACHE_SLOTS resident blocks (0 = off, the streamed path below). The reader
    // owns the slots and hands compute the slot to read over cb_slot, so a re-selected block costs no DRAM read.
    constexpr uint32_t KV_CACHE_SLOTS = get_compile_time_arg_val(28);
    constexpr uint32_t cb_k_cache = get_compile_time_arg_val(29);
    constexpr uint32_t cb_v_cache = get_compile_time_arg_val(30);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(31);
    // cb_slot depth = blocks the reader may run ahead of compute, so a miss's DRAM read overlaps the previous
    // block's math.
    constexpr uint32_t KV_CACHE_SLOT_DEPTH = get_compile_time_arg_val(32);
    // Slots compute may still be reading once reserve_back(cb_slot) returns; the busy scan is compiled out at depth 1.
    constexpr uint32_t KV_CACHE_INFLIGHT = KV_CACHE_SLOT_DEPTH > 1 ? KV_CACHE_SLOT_DEPTH - 1 : 1;
    // The victim search skips the in-flight slots, so it terminates only if some slot is never in flight.
    static_assert(
        KV_CACHE_SLOTS == 0 || KV_CACHE_SLOT_DEPTH == 1 || KV_CACHE_INFLIGHT < KV_CACHE_SLOTS,
        "cb_slot depth must leave at least one slot outside the in-flight set");

    // K/V use RuntimeTensorShape so T can vary without recompilation.
    constexpr auto q_args = TensorAccessorArgs<sparse_sdpa_msa::READER_CT_ARGS, 0>();
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
    // Per-device global position of this core's query row 0, patched at dispatch. Query rows >= straddle_row sit
    // straddle_jump positions further along: the boundary chip of a mid-slab (non-chunk-aligned) chunk start holds
    // the tail of one slab block and the head of its next one (jump 0 everywhere else).
    const uint32_t chunk_start_local = CAUSAL_MASK_ENABLED ? get_arg_val<uint32_t>(10) : 0;
    const uint32_t straddle_row = CAUSAL_MASK_ENABLED ? get_arg_val<uint32_t>(11) : 0;
    const uint32_t straddle_jump = CAUSAL_MASK_ENABLED ? get_arg_val<uint32_t>(12) : 0;
    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;

    Noc noc;
    // A CB handle is free to construct, so every build names all of them and uses only its own.
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_in), v_cb(cb_v_in), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    experimental::CB kreq_cb(cb_kreq), kack_cb(cb_kack);
    experimental::CB slot_cb(cb_slot), k_cache_cb(cb_k_cache), v_cache_cb(cb_v_cache);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    // Reader-internal scratch for one token's block-id row (reserved once, reused).
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    // Block-cache residency: slot s holds logical block kv_cache_bid[s] (sentinel = empty); misses fill the
    // slots round-robin. The cache CBs are reserved whole and never pushed, so slot addresses stay fixed. Any
    // slot assignment is correct -- the cache only changes where the bytes come from -- so the policy is free.
    // Sizes of 1 are placeholders (no zero-length arrays) for builds that never touch them.
    [[maybe_unused]] uint32_t kv_cache_bid[KV_CACHE_SLOTS > 0 ? KV_CACHE_SLOTS : 1];
    [[maybe_unused]] uint32_t kv_rr_next = 0;                  // round-robin cursor: the next victim candidate
    [[maybe_unused]] uint32_t kv_inflight[KV_CACHE_INFLIGHT];  // slots handed to compute in the last depth-1 blocks
    [[maybe_unused]] uint32_t kv_inflight_pos = 0;
    if constexpr (KV_CACHE_SLOTS > 0) {
        k_cache_cb.reserve_back(KV_CACHE_SLOTS * k_tiles_per_block);
        v_cache_cb.reserve_back(KV_CACHE_SLOTS * v_tiles_per_block);
        for (uint32_t s = 0; s < KV_CACHE_SLOTS; ++s) {
            kv_cache_bid[s] = sentinel;
        }
        for (uint32_t i = 0; i < KV_CACHE_INFLIGHT; ++i) {
            kv_inflight[i] = sentinel;
        }
    }

    // One gather request to the writer (cb_kreq page layout: sparse_sdpa_msa::kreq).
    auto post_kreq = [&](uint32_t block_id, uint32_t flags, uint32_t slot) {
        kreq_cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* rq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
        rq[kreq::BLOCK_ID] = block_id;
        rq[kreq::FLAGS] = flags;
        rq[kreq::SLOT] = slot;
        kreq_cb.push_back(1);
    };
    // Upper K/V tile halves of one block into L1 at byte offsets k_dst/v_dst of the destination CBs, then wait for
    // the writer's lower halves, so the whole block is resident when this returns.
    auto gather_upper_halves = [&](uint32_t k_tile0,
                                   uint32_t v_tile0,
                                   experimental::CB& k_dst_cb,
                                   experimental::CB& v_dst_cb,
                                   uint32_t k_dst,
                                   uint32_t v_dst) {
        sparse_sdpa_msa::TridRing ring{noc};  // K/V upper halves share one ring.
        for (uint32_t i = k_half; i < k_tiles_per_block; ++i) {
            ring.read(k, k_dst_cb, k_tile_bytes, k_tile0 + i, k_dst + i * k_tile_bytes);
        }
        for (uint32_t i = v_half; i < v_tiles_per_block; ++i) {
            ring.read(v, v_dst_cb, v_tile_bytes, v_tile0 + i, v_dst + i * v_tile_bytes);
        }
        ring.drain();           // this NoC's upper halves landed
        kack_cb.wait_front(1);  // writer's lower halves landed in the same L1
        kack_cb.pop_front(1);
    };

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
            const uint32_t p = chunk_start_local + tok + (tok >= straddle_row ? straddle_jump : 0);  // global position
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
            cp[ctrl::ACTIVE_BLOCKS] = n_active;
            cp[ctrl::DIAG_CHUNK] = diag_chunk;
            cp[ctrl::BOUNDARY_TILE] = boundary_tile;
            cp[ctrl::BOUNDARY_COL] = boundary_col;
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

            // Block-cyclic cache: remap the logical block id to its physical block before addressing (invP).
            // Addressing only — the sentinel search and the diagonal-block causal match stay on the logical id.
            // Identity for a natural-order cache (block_cyclic false).
            const uint32_t phys_block = tt::block_cyclic::
                logical_to_physical_page<block_cyclic, bc_chunk_local, bc_sp, bc_shard_stride_gap, bc_slab_stride_gap>(
                    block_id);
            uint32_t k_tile0 = k_batch_tile_offset + phys_block * k_tiles_per_block;
            uint32_t v_tile0 = v_batch_tile_offset + phys_block * v_tiles_per_block;
            if constexpr (n_kv > 1) {
                k_tile0 += kv_group * k_group_tile_stride;
                v_tile0 += kv_group * v_group_tile_stride;
            }
            const uint32_t last_flag = (chunk == n_active - 1) ? kreq::LAST : 0u;
            if constexpr (KV_CACHE_SLOTS > 0) {
                // Serve the block from its resident slot; a miss fills a victim slot the way the streamed path
                // fills cb_k_in/cb_v_in (this NoC the upper tile halves, the writer's NoC the lower ones).
                uint32_t slot = KV_CACHE_SLOTS;  // out of range = not resident
                for (uint32_t s = 0; s < KV_CACHE_SLOTS; ++s) {
                    if (kv_cache_bid[s] == block_id) {
                        slot = s;
                        break;
                    }
                }
                // Reserve before choosing a victim: once this returns compute holds at most depth-1 slot records,
                // so kv_inflight is exactly the set of slots it may still be reading.
                slot_cb.reserve_back(1);
                const bool miss = slot == KV_CACHE_SLOTS;
                if (miss) {
                    // Round-robin victim that skips the in-flight slots (terminates: INFLIGHT < SLOTS).
                    for (;;) {
                        const uint32_t cand = kv_rr_next;
                        kv_rr_next = (kv_rr_next + 1 == KV_CACHE_SLOTS) ? 0 : kv_rr_next + 1;
                        bool busy = false;
                        if constexpr (KV_CACHE_SLOT_DEPTH > 1) {
                            for (uint32_t i = 0; i < KV_CACHE_INFLIGHT; ++i) {
                                busy |= (kv_inflight[i] == cand);
                            }
                        }
                        if (!busy) {
                            slot = cand;
                            break;
                        }
                    }
                }
                // The writer hears about every miss (it fills the lower halves) and about the token's last block
                // (it then drains the output); hits in between stay silent.
                if (miss || last_flag) {
                    post_kreq(block_id, last_flag | (miss ? kreq::FETCH : 0u), slot);
                }
                if (miss) {
                    gather_upper_halves(
                        k_tile0,
                        v_tile0,
                        k_cache_cb,
                        v_cache_cb,
                        slot * k_tiles_per_block * k_tile_bytes,
                        slot * v_tiles_per_block * v_tile_bytes);
                    kv_cache_bid[slot] = block_id;
                }
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_cb.get_write_ptr()) = slot;
                slot_cb.push_back(1);
                if constexpr (KV_CACHE_SLOT_DEPTH > 1) {
                    kv_inflight[kv_inflight_pos] = slot;
                    kv_inflight_pos = (kv_inflight_pos + 1 == KV_CACHE_INFLIGHT) ? 0 : kv_inflight_pos + 1;
                }
            } else {
                // Streamed path: the reader reserves the whole block, the writer fills the lower half and the
                // reader the upper half; compute pops it after the chunk.
                k_cb.reserve_back(k_tiles_per_block);
                v_cb.reserve_back(v_tiles_per_block);
                post_kreq(block_id, kreq::FETCH | last_flag, 0);
                gather_upper_halves(k_tile0, v_tile0, k_cb, v_cb, 0, 0);
                k_cb.push_back(k_tiles_per_block);
                v_cb.push_back(v_tiles_per_block);
            }
        }

        ++tok;
        if constexpr (n_kv > 1) {
            if (tok == S) {
                tok = 0;
                ++kv_group;
                // A block id names different tiles in every KV group, so residency cannot carry across groups.
                if constexpr (KV_CACHE_SLOTS > 0) {
                    for (uint32_t s = 0; s < KV_CACHE_SLOTS; ++s) {
                        kv_cache_bid[s] = sentinel;
                    }
                }
            }
        }
    }
}
