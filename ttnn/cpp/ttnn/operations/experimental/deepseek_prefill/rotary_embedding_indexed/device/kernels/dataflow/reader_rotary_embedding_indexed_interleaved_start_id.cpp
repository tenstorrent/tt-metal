// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Forked from reader_rotary_embedding_llama_interleaved_start_id.cpp for KV-pad-aware indexed RoPE,
// on the Metal 2.0 named-arg API.
//
// The cos/sin caches are SP-sharded in block-cyclic order keyed by the per-device chunk size
// (chunk_local == Ht), so each device's shard already holds, in contiguous local-row order, the
// rope values for every global position that device will ever carry. This kernel only needs to
// derive WHERE in that shard the current chunk starts -- `update_idxt` -- exactly as the per-chip
// kv-cache writer does, then read cos/sin contiguously from there. The wrap of the boundary chip
// (older tokens finishing the current slab block, then newer tokens spilling into the next block)
// is absorbed by the shard layout, so the read stays contiguous.
//
// my_sp_coord / sp_factor are baked per-device compile-time args (each device's program is built for
// its own mesh coordinate). The per-call kv_actual_global reaches the reader one of two ways, selected
// by the HAS_METADATA compile-time flag (set by the op from whether a metadata tensor was supplied; both
// keep the value out of the program hash so one cached program is reused across chunks):
//   - scalar path: a common runtime arg (patched on cache hits by override_runtime_arguments).
//   - metadata path (HAS_METADATA): NoC-read of element [0] of a 1-element uint32 tensor bound as
//     tensor::metadata, so a captured trace advances the value via an in-place host update.
void kernel_main() {
    Noc noc;

    auto batch_start = get_arg(args::batch_start);
    auto batch_end = get_arg(args::batch_end);
    auto seq_t_start = get_arg(args::seq_t_start);
    auto seq_t_end = get_arg(args::seq_t_end);

    constexpr auto n_heads = get_arg(args::n_heads);
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr bool freq_per_head = get_arg(args::freq_per_head) == 1;
    constexpr auto cos_Ht = get_arg(args::cos_Ht);
    constexpr auto sin_Ht = get_arg(args::sin_Ht);
    constexpr auto rotary_Ht = get_arg(args::rotary_Ht);
    constexpr auto tile_height = get_arg(args::tile_height);
    // Per-device structural constants, baked when this device's program is built for its coordinate.
    constexpr auto my_sp_coord = get_arg(args::my_sp_coord);
    constexpr auto sp_factor = get_arg(args::sp_factor);

    // Metadata path: read kv_actual_global from element [0] of the 1-element uint32 tensor (4 bytes).
    // tensor::metadata / dfb::meta are bound only on the metadata program (coupled; host omits both).
    const uint32_t kv_actual_global = map_nullable_token(
        tensor::metadata,
        dfb::meta,
        [&](auto const& t, DFBBindingToken const& d) {
            const auto s_meta = TensorAccessor(t);
            DataflowBuffer dfb_meta(d);
            dfb_meta.reserve_back(1);
            uint32_t meta_l1_write_addr = dfb_meta.get_write_ptr();
            noc.async_read(s_meta, CoreLocalMem<uint32_t>(meta_l1_write_addr), 4, {.page_id = 0}, {});
            noc.async_read_barrier();
            // The metadata tensor lives at a FIXED DRAM address reused across every chunk/layer/rope call;
            // the host updates its contents in place each chunk. After the NoC writes the fresh value into
            // this core's dfb_meta L1 page, the RISC data cache may still hold the PREVIOUS chunk's value for
            // that L1 line: async_read_barrier orders the DMA but does NOT invalidate the RISC cache, and
            // `volatile` forces a load but still reads the cached line. Whether the line was evicted is
            // timing-dependent, so without this invalidate the read is intermittently STALE -> a wrong
            // rotation offset that compounds (the L61 metadata KV-PCC run-to-run non-determinism).
            // invalidate_l1_cache() forces a refetch of the freshly-DMA'd value.
            invalidate_l1_cache();
            CoreLocalMem<volatile uint32_t> meta(meta_l1_write_addr);
            const uint32_t value = meta[0];  // the 1-element tensor holds kv_actual_global directly
            dfb_meta.push_back(1);
            return value;
        },
        [&] { return get_arg(args::kv_actual_global); });

    // Convert the per-call kv_actual_global (tokens) to tiles.
    const uint32_t kv_actual_global_t = kv_actual_global / tile_height;
    // Derive this chip's tile-row offset into its (block-cyclic) cos/sin shard from the global
    // valid KV length. Ht == chunk_local_t (per-device new chunk in tiles); chunk_global == sp*Ht.
    // Identical math to the per-chip kv-cache writer's update_idxt -- see writer_update_padded_kv_cache.
    const uint32_t chunk_global_t = sp_factor * Ht;
    const uint32_t boundary_slab_idx = chunk_global_t == 0 ? 0 : kv_actual_global_t / chunk_global_t;
    const uint32_t boundary_chip = Ht == 0 ? 0 : (kv_actual_global_t / Ht) % sp_factor;
    const uint32_t boundary_offset_t = Ht == 0 ? 0 : kv_actual_global_t % Ht;
    // From the current slab base, chips before the boundary advance a full slab, the boundary chip
    // advances by its pad offset, and chips after it stay at the base.
    const uint32_t update_idxt =
        boundary_slab_idx * Ht +
        (my_sp_coord < boundary_chip ? Ht : (my_sp_coord == boundary_chip ? boundary_offset_t : 0));

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(tensor::input);
    const auto s1 = TensorAccessor(tensor::cos);
    const auto s2 = TensorAccessor(tensor::sin);
    const auto s3 = TensorAccessor(tensor::trans_mat);

    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_cos(dfb::cos);
    DataflowBuffer dfb_sin(dfb::sin);
    DataflowBuffer dfb_trans_mat(dfb::trans_mat);

    const uint32_t input_tile_bytes = dfb_input.get_entry_size();
    const uint32_t cos_tile_bytes = dfb_cos.get_entry_size();
    const uint32_t sin_tile_bytes = dfb_sin.get_entry_size();
    const uint32_t trans_mat_tile_bytes = dfb_trans_mat.get_entry_size();

    // Read transformation matrix in CB (only once, because it will be reused)
    dfb_trans_mat.reserve_back(onetile);
    uint32_t trans_mat_l1_write_addr = dfb_trans_mat.get_write_ptr();
    noc.async_read(s3, CoreLocalMem<uint32_t>(trans_mat_l1_write_addr), trans_mat_tile_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    dfb_trans_mat.push_back(onetile);

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        uint32_t sin_l1_write_addr = 0;
        uint32_t cos_l1_write_addr = 0;
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            dfb_sin.reserve_back(my_cos_sin_tiles);
            dfb_cos.reserve_back(my_cos_sin_tiles);
            sin_l1_write_addr = dfb_sin.get_write_ptr();
            cos_l1_write_addr = dfb_cos.get_write_ptr();
        }
#endif

        // To make sure the sin/cos row are read only once
        uint32_t sin_cos_row_cnt = 0;
        bool done_sin_cos = false;

        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
#if RELOAD_IMPL == 1
                dfb_sin.reserve_back(Wt);
                dfb_cos.reserve_back(Wt);
                uint32_t sin_l1_write_addr = dfb_sin.get_write_ptr();
                uint32_t cos_l1_write_addr = dfb_cos.get_write_ptr();
#endif

                dfb_input.reserve_back(Wt);
                uint32_t input_l1_write_addr = dfb_input.get_write_ptr();
                uint32_t input_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                // Offset the cos/sin source index by update_idxt: the input local tile `seq_tile`
                // is rotated by the value at shard row (update_idxt + seq_tile).
                const uint32_t rope_seq_tile = update_idxt + seq_tile;
                uint32_t cos_curr_idx;
                uint32_t sin_curr_idx;
                if constexpr (freq_per_head) {
                    cos_curr_idx = head_num * cos_Ht * Wt + rope_seq_tile * Wt;
                    sin_curr_idx = head_num * sin_Ht * Wt + rope_seq_tile * Wt;
                } else {
                    cos_curr_idx = rope_seq_tile * Wt;
                    sin_curr_idx = rope_seq_tile * Wt;
                }
                for (uint32_t j = 0; j < Wt; ++j) {
                    // Read input into CB
                    noc.async_read(
                        s0,
                        CoreLocalMem<uint32_t>(input_l1_write_addr),
                        input_tile_bytes,
                        {.page_id = input_curr_idx},
                        {});
                    input_curr_idx++;
                    input_l1_write_addr += input_tile_bytes;

                    if (!done_sin_cos) {
                        noc.async_read(
                            s2,
                            CoreLocalMem<uint32_t>(sin_l1_write_addr),
                            sin_tile_bytes,
                            {.page_id = sin_curr_idx},
                            {});
                        noc.async_read(
                            s1,
                            CoreLocalMem<uint32_t>(cos_l1_write_addr),
                            cos_tile_bytes,
                            {.page_id = cos_curr_idx},
                            {});
                        sin_curr_idx++;
                        cos_curr_idx++;
                        sin_l1_write_addr += sin_tile_bytes;
                        cos_l1_write_addr += cos_tile_bytes;
                    }
                }

                noc.async_read_barrier();
                dfb_input.push_back(Wt);
#if RELOAD_IMPL == 1
                dfb_sin.push_back(Wt);
                dfb_cos.push_back(Wt);
#else
                if (!done_sin_cos) {
                    dfb_sin.push_back(Wt);
                    dfb_cos.push_back(Wt);
                    // Update sin_cos_row_cnt
                    sin_cos_row_cnt++;
                    if (sin_cos_row_cnt == my_rotary_seq_tiles) {
                        done_sin_cos = true;
                    }
                }
#endif
            }
        }
    }
}
