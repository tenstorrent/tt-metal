// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// onorm reader (NCRISC / NoC0).
//
// ALL DRAM reads of the op live here: `weight` + the reduce scaler once per
// core, then per token-block the `o` stream (head-major) and the `gate` stream
// (flat, pre-sigmoid).  Reads issued on NoC1 measure ~4.8x slower than on NoC0,
// so the writer never reads anything — see op_design.md §6.
//
// Both per-block streams are `dm_block_tiles`-granular groups with ONE
// noc_async_read_barrier per group: `dm_block_tiles` reads are in flight at once
// so the transfers pipeline instead of paying a full DRAM round trip per tile.
// The group size is the DM_BLOCK_TILES knob (a compile-time arg), never a
// literal.
//
// HELPER USAGE: the only kernel_lib helper that applies to a dataflow kernel
// here is dataflow_kernel_lib::prepare_reduce_scaler (used, in its
// pool-type-aware form).  Plain interleaved-DRAM tile streaming has no
// kernel_lib helper, so the TensorAccessor + noc_async_read pair below is raw by
// necessity, not by preference.  mcast_pipe.hpp was considered and rejected:
// phase 1 shares no operand across cores (every stream is disjoint per core),
// so there is no multicast to perform.

#include "api/dataflow/dataflow_api.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

// Read `count` consecutive tile ids into `cb`, in groups of `group` tiles with
// one barrier per group.
template <uint32_t cb, uint32_t group, uint32_t page_bytes, typename Accessor>
FORCE_INLINE void stream_tiles(const Accessor& acc, uint32_t first_tile, uint32_t count) {
    uint32_t done = 0;
    while (done < count) {
        const uint32_t remaining = count - done;
        const uint32_t n = remaining < group ? remaining : group;
        cb_reserve_back(cb, n);
        const uint32_t l1_write_addr = get_write_ptr(cb);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(acc.get_noc_addr(first_tile + done + i), l1_write_addr + i * page_bytes, page_bytes);
        }
        noc_async_read_barrier();  // ONE barrier for `n` reads
        cb_push_back(cb, n);
        done += n;
    }
}

}  // namespace

void kernel_main() {
    // CB slot map — injected as preprocessor defines from the ONE host-side
    // source of truth (`_CB_SLOTS` in onorm_program_descriptor.py).
    constexpr uint32_t cb_o_tiles = ONORM_CB_O_TILES;
    constexpr uint32_t cb_gate_tiles = ONORM_CB_GATE_TILES;
    constexpr uint32_t cb_weight = ONORM_CB_WEIGHT;
    constexpr uint32_t cb_scaler = ONORM_CB_SCALER;

    // --- Blocking Model parameters (compile-time; one source of truth on host) ---
    constexpr uint32_t v_tiles = get_compile_time_arg_val(0);              // V / TILE_W
    constexpr uint32_t tokens_per_block = get_compile_time_arg_val(1);     // TOKENS_PER_BLOCK
    constexpr uint32_t flat_tiles = get_compile_time_arg_val(2);           // FLAT / TILE_W
    constexpr uint32_t tile_rows_per_block = get_compile_time_arg_val(3);  // TOKENS_PER_BLOCK / TILE_H
    constexpr uint32_t blocks_per_batch = get_compile_time_arg_val(4);     // ceil(T / TOKENS_PER_BLOCK)
    constexpr uint32_t tokens_total = get_compile_time_arg_val(5);         // T
    constexpr uint32_t token_tile_rows = get_compile_time_arg_val(6);      // Tt = ceil(T / TILE_H)
    constexpr uint32_t dm_block_tiles = get_compile_time_arg_val(7);       // DM_BLOCK_TILES
    constexpr uint32_t inv_v_bits = get_compile_time_arg_val(8);           // fp32 bits of 1/V
    constexpr uint32_t page_bytes = get_compile_time_arg_val(9);

    constexpr auto o_args = TensorAccessorArgs<10>();
    constexpr auto gate_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();

    // Derived per-block tile counts — never restated literals.
    constexpr uint32_t o_tiles_per_block = tokens_per_block * v_tiles;
    constexpr uint32_t flat_tiles_per_block = tile_rows_per_block * flat_tiles;

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t weight_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_block = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks = get_arg_val<uint32_t>(4);

    const auto o_acc = TensorAccessor(o_args, o_addr, page_bytes);
    const auto gate_acc = TensorAccessor(gate_args, gate_addr, page_bytes);
    const auto weight_acc = TensorAccessor(weight_args, weight_addr, page_bytes);

    // ---- once per core: the reuse-shared weight row (v_tiles tiles, never popped) ----
    cb_reserve_back(cb_weight, v_tiles);
    {
        MaybeDeviceZoneScope("onorm_read_weight");
        const uint32_t l1_write_addr = get_write_ptr(cb_weight);
        for (uint32_t i = 0; i < v_tiles; ++i) {
            noc_async_read(weight_acc.get_noc_addr(i), l1_write_addr + i * page_bytes, page_bytes);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_weight, v_tiles);

    // ---- once per core: the reduce scaler = 1/V (never popped) ----
    // The divisor is explicit and host-supplied; it is never derived from tile
    // geometry.  Pool-type-aware overload: SUM + REDUCE_ROW picks the fill the
    // reduce LLK actually consumes.
    union {
        uint32_t u;
        float f;
    } inv_v;
    inv_v.u = inv_v_bits;
    {
        MaybeDeviceZoneScope("onorm_fill_scaler");
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            inv_v.f);
    }

    // ---- this core's token-blocks ----
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t bi = start_block + blk;
        const uint32_t b = bi / blocks_per_batch;  // batch
        const uint32_t r = bi % blocks_per_batch;  // token-block within the batch

        // `o` is tiled over (HV, V), so its token axis is UN-padded.
        const uint32_t o_first_tile = (b * tokens_total + r * tokens_per_block) * v_tiles;
        {
            MaybeDeviceZoneScope("onorm_read_o");
            stream_tiles<cb_o_tiles, dm_block_tiles, page_bytes>(o_acc, o_first_tile, o_tiles_per_block);
        }

        // `gate` is tiled over (T, FLAT), so its token axis IS tile-padded (Tt).
        const uint32_t gate_first_tile = (b * token_tile_rows + r * tile_rows_per_block) * flat_tiles;
        {
            MaybeDeviceZoneScope("onorm_read_gate");
            stream_tiles<cb_gate_tiles, dm_block_tiles, page_bytes>(gate_acc, gate_first_tile, flat_tiles_per_block);
        }
    }
}
