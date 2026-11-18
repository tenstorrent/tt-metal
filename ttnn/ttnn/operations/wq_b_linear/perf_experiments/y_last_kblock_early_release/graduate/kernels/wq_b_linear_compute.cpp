// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// y block (block_m_tiles x core_n_tiles) = x (resident, K-blocked) @ W (streamed, K-blocked),
// accumulated in place in cb_y_tiles via packer L1 accumulation. The last K-block runs column group by column group
// and releases each group to the y store as soon as it is packed.
// Regime R4 (has_slices): a core may also own a K-slice (slice_k_blocks K-blocks from slice_kb_start) of one
// leftover column. That matmul runs nested in the main one, after main K-block slice_after_block, reading x
// through cb_x_slice (an alias of cb_x_resident with its own FIFO pointers) and packing to cb_partial. The owner
// (num_landed > 0), after main K-block combine_after_block, writes the leftover column into cb_y_tiles after
// its own block as own partial + landed partials in slice order.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

using namespace compute_kernel_lib;

// Raw-LLK last K-block (bypasses compute_kernel_lib::matmul_block): with caller_owns_pack_target the helper pushes
// nothing until the caller's single push after the whole K-loop, and its L1-accumulation switch is keyed on the block
// index within one call, so a separate one-block helper call would overwrite the earlier blocks' partials. This loop
// issues the helper's LLK sequence for that block (same subblocks, same per-tile k order, L1-accumulating pack) with
// column groups outermost, and pushes sb_w tiles after each group but the last. Each push moves the write pointer
// sb_w tiles on, so later packs subtract the tiles pushed so far. Returns the tiles pushed.
ALWI uint32_t last_k_block_by_column(
    CircularBuffer& x_buf,
    CircularBuffer& w_buf,
    CircularBuffer& y_buf,
    uint32_t block_m_tiles,
    uint32_t core_n_tiles,
    uint32_t sb_h,
    uint32_t sb_w,
    uint32_t k_block_tiles,
    bool first_k_block) {
    const uint32_t cb_x = x_buf.get_cb_id();
    const uint32_t cb_w = w_buf.get_cb_id();
    const uint32_t cb_y = y_buf.get_cb_id();
    if (first_k_block) {
        ckernel::matmul_block_init(cb_x, cb_w, false, sb_w, sb_h, k_block_tiles);
    }
    x_buf.wait_front(block_m_tiles * k_block_tiles);
    w_buf.wait_front(k_block_tiles * core_n_tiles);
    PACK((llk_pack_reconfig_l1_acc(first_k_block ? 0 : 1)));
    uint32_t pushed = 0;
    for (uint32_t n0 = 0; n0 < core_n_tiles; n0 += sb_w) {
        for (uint32_t m0 = 0; m0 < block_m_tiles; m0 += sb_h) {
            tile_regs_acquire();
#ifndef SKIP_COMPUTE
            for (uint32_t k = 0; k < k_block_tiles; ++k) {
                ckernel::matmul_block(
                    cb_x, cb_w, m0 * k_block_tiles + k, n0 + k * core_n_tiles, 0, false, sb_w, sb_h, k_block_tiles);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < sb_h; ++r) {
                for (uint32_t c = 0; c < sb_w; ++c) {
                    pack_tile<true>(r * sb_w + c, cb_y, (m0 + r) * core_n_tiles + n0 + c - pushed);
                }
            }
            tile_regs_release();
        }
        if (n0 + sb_w < core_n_tiles) {
            y_buf.push_back(sb_w);
            pushed += sb_w;
        }
    }
    x_buf.pop_front(block_m_tiles * k_block_tiles);
    w_buf.pop_front(k_block_tiles * core_n_tiles);
    return pushed;
}

template <typename PostKBlockFn>
ALWI void accumulate_block(
    CircularBuffer& in0,
    CircularBuffer& in1,
    CircularBuffer& out,
    const MatmulBlockShape& shape,
    PostKBlockFn post_k_block = {}) {
    matmul_block<
        false,
        true,
        LastBlockTarget::Interm,
        OutputCBLayout::TileRowMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::WaitAndPopPerKBlock,
        NoPostCompute,
        NoPreKBlock,
        PostKBlockFn,
        0,
        NoKBlockInnerDimFn,
        NoIn0Source,
        NoIn1BaseOffset,
        true>(in0, in1, out, out, shape, {}, {}, 0, 0, post_k_block);
}

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_stream = get_compile_time_arg_val(1);
    constexpr uint32_t cb_y_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t sb_h = get_compile_time_arg_val(5);
    constexpr uint32_t sb_w = get_compile_time_arg_val(6);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(8);
    constexpr bool has_slices = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t cb_x_slice = get_compile_time_arg_val(10);
    constexpr uint32_t cb_w_slice = get_compile_time_arg_val(11);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(12);
    constexpr uint32_t cb_landing = get_compile_time_arg_val(13);

    const uint32_t slice_kb_start = get_arg_val<uint32_t>(0);
    const uint32_t slice_k_blocks = get_arg_val<uint32_t>(1);
    const uint32_t slice_after_block = get_arg_val<uint32_t>(2);
    const uint32_t num_landed = get_arg_val<uint32_t>(3);
    const uint32_t combine_after_block = get_arg_val<uint32_t>(4);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_stream, cb_y_tiles);

    CircularBuffer x_buf(cb_x_resident);
    CircularBuffer w_buf(cb_w_stream);
    CircularBuffer y_buf(cb_y_tiles);

    constexpr uint32_t y_block_tiles = block_m_tiles * core_n_tiles;
    const uint32_t y_tiles = y_block_tiles + (num_landed > 0 ? block_m_tiles : 0);
    const auto main_shape =
        MatmulBlockShape::of(block_m_tiles / sb_h, core_n_tiles / sb_w, sb_h, sb_w, k_block_tiles, num_k_blocks);

    y_buf.reserve_back(y_tiles);
    uint32_t pushed = 0;
    {
        MaybeDeviceZoneScope("compute_first_kblock_wait");
        x_buf.wait_front(block_m_tiles * k_block_tiles);
        w_buf.wait_front(k_block_tiles * core_n_tiles);
    }
    MaybeDeviceZoneScope("compute_matmul");
    if constexpr (has_slices) {
        auto restore_main = [&]() {
            reconfig_data_format(cb_w_stream, cb_x_resident);
            PACK((pack_reconfig_data_format(cb_y_tiles)));
            ckernel::matmul_block_init(cb_x_resident, cb_w_stream, false, sb_w, sb_h, k_block_tiles);
        };
        auto slice_matmul = [&]() {
            CircularBuffer xs_buf(cb_x_slice);
            CircularBuffer ws_buf(cb_w_slice);
            CircularBuffer partial_buf(cb_partial);
            constexpr uint32_t k_block_x_tiles = block_m_tiles * k_block_tiles;
            if (slice_kb_start > 0) {
                xs_buf.wait_front(slice_kb_start * k_block_x_tiles);
                xs_buf.pop_front(slice_kb_start * k_block_x_tiles);
            }
            partial_buf.reserve_back(block_m_tiles);
            accumulate_block(
                xs_buf,
                ws_buf,
                partial_buf,
                MatmulBlockShape::of(1, 1, block_m_tiles, 1, k_block_tiles, slice_k_blocks),
                NoPostKBlock{});
            partial_buf.push_back(block_m_tiles);
        };
        auto combine_leftover = [&]() {
            // matmul_block leaves packer L1 accumulation on; the chain below assumes it starts off.
            pack_reconfig_l1_acc(0);
            DataflowBuffer partial_dfb(cb_partial);
            DataflowBuffer landing_dfb(cb_landing);
            DataflowBuffer y_dfb(cb_y_tiles);
            eltwise_chain(
                IterationShape::tiles(block_m_tiles),
                CopyTile<input(cb_partial, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)>{
                    partial_dfb},
                PackTile<output(
                    cb_y_tiles,
                    ReservePolicy::None,
                    PushPolicy::None,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Offset)>{y_dfb, y_block_tiles});
            // Landed tile t of slice s sits at t * num_landed + s - 1: one accumulating pass per output tile.
            landing_dfb.wait_front(num_landed * block_m_tiles);
            for (uint32_t t = 0; t < block_m_tiles; ++t) {
                eltwise_chain(
                    IterationShape::tiles(num_landed),
                    CopyTile<input(
                        cb_landing,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset)>{landing_dfb, t * num_landed},
                    PackTile<output(
                        cb_y_tiles,
                        ReservePolicy::None,
                        PushPolicy::None,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset,
                        DestAccumulation::Disabled,
                        L1Accumulation::AddToExisting)>{y_dfb, y_block_tiles + t});
            }
            landing_dfb.pop_front(num_landed * block_m_tiles);
        };
        // Both stages run between main K-blocks; the host orders slice_after_block <= combine_after_block.
        auto between_k_blocks = [&](uint32_t block, uint32_t, bool) {
            if (slice_k_blocks > 0 && block == slice_after_block) {
                slice_matmul();
                restore_main();
            }
            if (num_landed > 0 && block == combine_after_block) {
                combine_leftover();
                restore_main();
            }
        };
        accumulate_block(x_buf, w_buf, y_buf, main_shape, between_k_blocks);
    } else {
        if constexpr (num_k_blocks > 1) {
            accumulate_block(
                x_buf,
                w_buf,
                y_buf,
                MatmulBlockShape::of(
                    block_m_tiles / sb_h, core_n_tiles / sb_w, sb_h, sb_w, k_block_tiles, num_k_blocks - 1),
                NoPostKBlock{});
        }
        pushed = last_k_block_by_column(
            x_buf, w_buf, y_buf, block_m_tiles, core_n_tiles, sb_h, sb_w, k_block_tiles, num_k_blocks == 1);
    }
    y_buf.push_back(y_tiles - pushed);
}
