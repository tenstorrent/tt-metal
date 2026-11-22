// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_msa writer: co-gather lower K/V tile halves (K in the recipe's K^T layout), write row-major output, and
// build the recipe's persistent compute tiles (reduce scaler, column identity).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar_zeroed
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"  // per-NoC trid-ring (K_TRID_RING knob)
#include "block_cyclic_remap.hpp"      // tt::block_cyclic::logical_to_physical_page (block-cyclic cache remap)
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_msa_common.hpp"  // kreq record
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"

namespace kreq = sparse_sdpa_msa::kreq;

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed

void kernel_main() {
    namespace ct = sparse_sdpa_msa::writer_ct;
    constexpr uint32_t H_logical = get_compile_time_arg_val(ct::H_LOGICAL);
    constexpr uint32_t S = get_compile_time_arg_val(ct::S);
    constexpr uint32_t n_kv = get_compile_time_arg_val(ct::N_KV);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(ct::ROW_BYTES);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(ct::BLOCK_TILES);
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(ct::K_TILES_PER_BLOCK);
    constexpr uint32_t v_tiles_per_block = get_compile_time_arg_val(ct::V_TILES_PER_BLOCK);
    constexpr uint32_t k_half = get_compile_time_arg_val(ct::K_HALF);
    constexpr uint32_t v_half = get_compile_time_arg_val(ct::V_HALF);

    namespace cb = sparse_sdpa_msa::recipe_cb;
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(ct::K_TILE_BYTES);
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(ct::V_TILE_BYTES);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(ct::D_TILES);
    constexpr uint32_t k_tiles = k_tiles_per_block / d_tiles;

    // Block-cyclic ("slab") cache remap in BLOCK units; same constants as the reader (see its comment).
    constexpr bool block_cyclic = get_compile_time_arg_val(ct::BLOCK_CYCLIC) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(ct::BC_CHUNK_LOCAL);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(ct::BC_SP);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(ct::BC_SHARD_STRIDE_GAP);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(ct::BC_SLAB_STRIDE_GAP);
    // Per-core K/V block cache: when on, the writer fills the lower tile halves of each missed block into the
    // reader's victim slot of the K / V CBs (offsets from their bases); hits never reach it.
    constexpr uint32_t KV_CACHE_SLOTS = get_compile_time_arg_val(ct::KV_CACHE_SLOTS);
    constexpr auto out_args = TensorAccessorArgs<ct::COUNT, 0>();
    // K/V use RuntimeTensorShape so T can vary without recompilation.
    constexpr auto k_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto v_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_start = get_arg_val<uint32_t>(1);
    const uint32_t work_count = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t k_batch_tile_offset = get_arg_val<uint32_t>(5);  // cache_batch_idx slot offset (0 if not indexed)
    const uint32_t v_batch_tile_offset = get_arg_val<uint32_t>(6);
    uint32_t k_group_tile_stride = 0;
    uint32_t v_group_tile_stride = 0;
    if constexpr (n_kv > 1) {
        k_group_tile_stride = get_arg_val<uint32_t>(7);
        v_group_tile_stride = get_arg_val<uint32_t>(8);
    }

    Noc noc;
    // The writer never pushes the K / V CBs, so its views of their write pointers stay at their bases: the streamed
    // block (one entry) and slot 0.
    experimental::CB out_cb(cb::out_rm), kreq_cb(cb::kreq), kack_cb(cb::kack);
    const uint32_t k_base = experimental::CB(recipe_ring_cb::k).get_write_ptr();
    const uint32_t v_base = experimental::CB(recipe_ring_cb::v).get_write_ptr();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);

    // The recipe's reduce scaler and column identity (as the dense recipe reader, reader_recipe.cpp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        recipe_ring_cb::identity_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar_zeroed(CircularBuffer(recipe_ring_cb::col_identity), one_bf16_packed);

    uint32_t tok = work_start;
    uint32_t kv_group = 0;
    if constexpr (n_kv > 1) {
        kv_group = work_start / S;
        tok = work_start - kv_group * S;
    }
    // Lower K/V tile halves of one block: DRAM tiles [k_tile0, k_tile0 + k_half) -> L1 at k_dst (K^T positions) /
    // v_dst + i * tile. Block-cyclic cache: the logical block id (from the reader) is remapped to its physical block
    // (invP); identity for a natural-order cache (block_cyclic false).
    auto gather_lower_halves = [&](uint32_t block_id, uint32_t k_dst, uint32_t v_dst) {
        const uint32_t phys_block = tt::block_cyclic::
            logical_to_physical_page<block_cyclic, bc_chunk_local, bc_sp, bc_shard_stride_gap, bc_slab_stride_gap>(
                block_id);
        uint32_t k_tile0 = k_batch_tile_offset + phys_block * k_tiles_per_block;
        uint32_t v_tile0 = v_batch_tile_offset + phys_block * v_tiles_per_block;
        if constexpr (n_kv > 1) {
            k_tile0 += kv_group * k_group_tile_stride;
            v_tile0 += kv_group * v_group_tile_stride;
        }
        sparse_sdpa_msa::TridRing ring{noc};  // K/V lower halves share one ring.
        for (uint32_t i = 0; i < k_half; ++i) {
            ring.read(
                k,
                k_tile_bytes,
                k_tile0 + i,
                k_dst + sparse_sdpa_msa::kt_tile_position(i, d_tiles, k_tiles) * k_tile_bytes);
        }
        for (uint32_t i = 0; i < v_half; ++i) {
            ring.read(v, v_tile_bytes, v_tile0 + i, v_dst + i * v_tile_bytes);
        }
        ring.drain();
        kack_cb.reserve_back(1);
        kack_cb.push_back(1);
    };

    for (uint32_t work = 0; work < work_count; ++work) {
        // Serve the reader's gather requests for this token until its LAST, fetching the lower K/V tile halves of
        // every block flagged FETCH into the streamed block buffer or the named cache slot.
        bool last = false;
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t block_id, flags, slot;
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                block_id = rq[kreq::BLOCK_ID];
                flags = rq[kreq::FLAGS];
                slot = rq[kreq::SLOT];
            }
            kreq_cb.pop_front(1);
            last = (flags & kreq::LAST) != 0;
            if (flags & kreq::FETCH) {
                gather_lower_halves(
                    block_id,
                    k_base + slot * k_tiles_per_block * k_tile_bytes,
                    v_base + slot * v_tiles_per_block * v_tile_bytes);
            }
        }

        out_cb.wait_front(block_tiles);  // one untilized [H, v_dim] block
        for (uint32_t h = 0; h < H_logical; ++h) {
            // Row h at CB byte offset h*row_bytes maps to output head h within this KV group.
            uint32_t out_head = h;
            if constexpr (n_kv > 1) {
                out_head += kv_group * H_logical;
            }
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = out_head * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);

        ++tok;
        if constexpr (n_kv > 1) {
            if (tok == S) {
                tok = 0;
                ++kv_group;
            }
        }
    }
}
