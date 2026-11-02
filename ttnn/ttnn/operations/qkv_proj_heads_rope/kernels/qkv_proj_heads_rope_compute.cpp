// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope compute: per block, project_block (matmul_block over K' = hidden + bias tile into
// Float32 L1-accumulated partials), then the epilogue per valid row: rotate_block on q/k cores
// (two DEST-accumulated FPU muls per output tile), convert_block (copy + pack to bf16) on v cores.
// Output rows alternate between cb_out_tiles (writer, NoC1) and cb_out_tiles_noc0 (reader, NoC0) when
// out_write_nocs == 2.
//
// Raw calls: pack_reconfig_l1_acc(0) after each block - no helper exposes the L1-acc disable
// (matmul_block leaves it enabled after the last K-block); the epilogue chains' setup (reconfig +
// mul_init / copy_init) once per block, which InitReconfigOwner::Caller hands to the caller so it is not
// repeated per row x head.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

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
    constexpr uint32_t out_write_nocs = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out_tiles_noc0 = get_compile_time_arg_val(13);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(14);

    const uint32_t core_rows = get_arg_val<uint32_t>(0);
    const bool is_rotary = get_arg_val<uint32_t>(1) != 0;

    using namespace compute_kernel_lib;

    constexpr uint32_t n_tiles = heads_per_core * tiles_per_head;
    constexpr uint32_t partials_tiles = block_m * n_tiles;
    constexpr uint32_t table_tiles_per_row = 2 * tiles_per_head;
    static_assert(block_m % sb_h == 0 && n_tiles % sb_w == 0);
    static_assert(tiles_per_head == 2, "rotate_half as a tile swap needs D/2 == one tile");
    static_assert(out_write_nocs == 1 || out_write_nocs == 2);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_activation_block, cb_weight_block, cb_matmul_partials);

    CircularBuffer activation(cb_activation_block);
    CircularBuffer weight(cb_weight_block);
    CircularBuffer partials(cb_matmul_partials);
    DataflowBuffer partials_dfb(cb_matmul_partials);
    DataflowBuffer table_dfb(cb_rotary_table);
    DataflowBuffer out_dfb(cb_out_tiles);

    constexpr auto shape = MatmulBlockShape::of(block_m / sb_h, n_tiles / sb_w, sb_h, sb_w, k_block, num_k_blocks);

    // rotate_block: walk grid(2 out tiles, 2 terms). A = partials of one head (Row: base + term),
    // B = the row's table slot [cos_lo, -sin_lo, sin_hi, cos_hi] (Block: base + 2*out + term), summed in DEST.
    using RotaryMulAcc = BinaryFpu<
        BinaryFpuOp::Mul,
        input(
            cb_matmul_partials,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Row,
            DataFormatReconfig::Disabled,
            TileAddressing::Offset),
        input(
            cb_rotary_table,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Block,
            DataFormatReconfig::Disabled,
            TileAddressing::Offset),
        Dst::D0,
        DestAccumulation::PerRow>;
    using RotaryPack = PackTile<output(
        cb_out_tiles,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Disabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;
    using RotaryPackNoc0 = PackTile<output(
        cb_out_tiles_noc0,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Disabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;
    // convert_block: Float32 partials row -> bf16 output row.
    using ConvertCopy = CopyTile<input(
        cb_matmul_partials,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Disabled,
        TileAddressing::Offset)>;
    using ConvertPack =
        PackTile<output(cb_out_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled)>;
    using ConvertPackNoc0 =
        PackTile<output(cb_out_tiles_noc0, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled)>;

    // Epilogue of one valid row; the pack elements are bound to that row's output CB.
    auto epilogue_row = [&](uint32_t row, auto rotary_pack, auto convert_pack) {
        const uint32_t row_base = row * n_tiles;
        if (is_rotary) {
            const uint32_t table_base = (row % seq_tiles) * table_tiles_per_row;
            for (uint32_t head = 0; head < heads_per_core; ++head) {
                eltwise_chain<InitReconfigOwner::Caller>(
                    IterationShape::grid(tiles_per_head, 2),
                    RotaryMulAcc{partials_dfb, table_dfb, row_base + head * tiles_per_head, table_base},
                    rotary_pack);
            }
        } else {
            eltwise_chain<InitReconfigOwner::Caller>(
                IterationShape::tiles(n_tiles), ConvertCopy{partials_dfb, row_base}, convert_pack);
        }
    };

    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;

        // project_block: the whole nominal block accumulates in place (caller-owned pack target).
        {
            MaybeDeviceZoneScope("compute_partials_reserve");
            partials.reserve_back(partials_tiles);
        }
        {
            // Occupancy: matmul_block waits per K-block on x / W inside the helper.
            MaybeDeviceZoneScope("compute_matmul");
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
        }
        partials.push_back(partials_tiles);
        pack_reconfig_l1_acc(0);

        // The epilogue chains' one-time setup, once per block (InitReconfigOwner::Caller): both output
        // CBs are bf16, so one pack reconfig serves either.
        if (is_rotary) {
            reconfig_data_format(cb_matmul_partials, cb_rotary_table);
            mul_init(cb_matmul_partials, cb_rotary_table, true);
        } else {
            reconfig_data_format_srca(cb_matmul_partials);
            copy_init(cb_matmul_partials);
        }
        pack_reconfig_data_format(cb_out_tiles);

        // Table slot j holds the block's sequence position (block start + j) % seq_tiles.
        const uint32_t table_tiles = (rows < seq_tiles ? rows : seq_tiles) * table_tiles_per_row;
        {
            MaybeDeviceZoneScope("compute_epilogue_wait");
            if (is_rotary) {
                cb_wait_front(cb_rotary_table, table_tiles);
            }
            partials.wait_front(partials_tiles);
        }
        // Occupancy: the pack elements reserve cb_out_tiles / cb_out_tiles_noc0 per row.
        MaybeDeviceZoneScope("compute_epilogue");
        for (uint32_t row = 0; row < rows; ++row) {
            if constexpr (out_write_nocs == 2) {
                if (row & 1) {
                    DataflowBuffer out_noc0_dfb(cb_out_tiles_noc0);
                    epilogue_row(row, RotaryPackNoc0{out_noc0_dfb}, ConvertPackNoc0{out_noc0_dfb});
                    continue;
                }
            }
            epilogue_row(row, RotaryPack{out_dfb}, ConvertPack{out_dfb});
        }
        partials.pop_front(partials_tiles);
        if (is_rotary) {
            cb_pop_front(cb_rotary_table, table_tiles);
        }
    }
}
