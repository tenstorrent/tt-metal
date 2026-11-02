// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope compute: per block, project_block (matmul_block over K' = hidden + bias tile into
// Float32 L1-accumulated partials), then the epilogue per valid row: rotate_block on q/k cores
// (two DEST-accumulated FPU muls per output tile), convert_block (copy + pack to bf16) on v cores.
//
// Raw call: pack_reconfig_l1_acc(0) after each block - no helper exposes the L1-acc disable
// (matmul_block leaves it enabled after the last K-block).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_activation_block = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_block = get_compile_time_arg_val(1);
    constexpr uint32_t cb_matmul_partials = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rotary_table = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t block_m = get_compile_time_arg_val(5);
    constexpr uint32_t heads_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(7);
    constexpr uint32_t k_block = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t sb_h = get_compile_time_arg_val(10);
    constexpr uint32_t sb_w = get_compile_time_arg_val(11);

    const uint32_t core_rows = get_arg_val<uint32_t>(0);
    const bool is_rotary = get_arg_val<uint32_t>(1) != 0;

    using namespace compute_kernel_lib;

    constexpr uint32_t n_tiles = heads_per_core * tiles_per_head;
    constexpr uint32_t partials_tiles = block_m * n_tiles;
    constexpr uint32_t table_tiles_per_row = 2 * tiles_per_head;
    static_assert(block_m % sb_h == 0 && n_tiles % sb_w == 0);
    static_assert(tiles_per_head == 2, "rotate_half as a tile swap needs D/2 == one tile");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_activation_block, cb_weight_block, cb_matmul_partials);

    CircularBuffer activation(cb_activation_block);
    CircularBuffer weight(cb_weight_block);
    CircularBuffer partials(cb_matmul_partials);
    DataflowBuffer partials_dfb(cb_matmul_partials);
    DataflowBuffer table_dfb(cb_rotary_table);
    DataflowBuffer out_dfb(cb_out_tiles);

    constexpr auto shape = MatmulBlockShape::of(block_m / sb_h, n_tiles / sb_w, sb_h, sb_w, k_block, num_k_blocks);

    // rotate_block: walk grid(2 out tiles, 2 terms). A = partials of one head (Row: base + term),
    // B = table row [cos_lo, -sin_lo, sin_hi, cos_hi] (Block: 2*out + term), products summed in DEST.
    using RotaryMulAcc = BinaryFpu<
        BinaryFpuOp::Mul,
        input(
            cb_matmul_partials,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Row,
            DataFormatReconfig::Enabled,
            TileAddressing::Offset),
        input(cb_rotary_table, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
        Dst::D0,
        DestAccumulation::PerRow>;
    using RotaryPack = PackTile<output(
        cb_out_tiles,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;
    // convert_block: Float32 partials row -> bf16 output row.
    using ConvertCopy = CopyTile<input(
        cb_matmul_partials,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset)>;
    using ConvertPack = PackTile<output(cb_out_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd)>;

    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;

        // project_block: the whole nominal block accumulates in place (caller-owned pack target).
        partials.reserve_back(partials_tiles);
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
            NoPostKBlock,
            0,
            NoKBlockInnerDimFn,
            NoIn0Source,
            NoIn1BaseOffset,
            true>(activation, weight, partials, partials, shape, {}, {}, n_tiles, n_tiles);
        partials.push_back(partials_tiles);
        pack_reconfig_l1_acc(0);

        partials.wait_front(partials_tiles);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t row_base = row * n_tiles;
            if (is_rotary) {
                cb_wait_front(cb_rotary_table, table_tiles_per_row);
                for (uint32_t head = 0; head < heads_per_core; ++head) {
                    eltwise_chain(
                        IterationShape::grid(tiles_per_head, 2),
                        RotaryMulAcc{partials_dfb, table_dfb, row_base + head * tiles_per_head},
                        RotaryPack{out_dfb});
                }
                cb_pop_front(cb_rotary_table, table_tiles_per_row);
            } else {
                eltwise_chain(
                    IterationShape::tiles(n_tiles), ConvertCopy{partials_dfb, row_base}, ConvertPack{out_dfb});
            }
        }
        partials.pop_front(partials_tiles);
    }
}
