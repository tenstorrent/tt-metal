// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope compute: per block, project_block (matmul_block over K' = hidden + bias tile into
// Float32 L1-accumulated partials), then the epilogue per valid row: rotate_block on q/k cores
// (two DEST-accumulated FPU muls per output tile), convert_block (copy + pack to bf16) on v cores.
// Output rows alternate between cb_out_tiles (writer, NoC1) and cb_out_tiles_noc0 (reader, NoC0) when
// out_write_nocs == 2.
//
// EXPERIMENT (epilogue_overlap): the last K-block runs per output row with raw LLK, and each row's
// epilogue (and its push to the store path) follows that row's final K-block, so the reader/writer stores
// overlap the rest of the last K-block instead of bursting after the whole K-loop.
// Raw LLK bypasses compute_kernel_lib::matmul_block for the last K-block: the helper's L1-acc pack config
// is llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1), so a per-row call carrying only the last K-block
// (num_k_blocks = 1) would overwrite the partials K-blocks 0..n-2 accumulated, and it has no per-row
// hook between subblock packs to run the epilogue. kLag (CT arg 15) = rows of last-K-block matmul run
// ahead of the epilogue, so the epilogue's unpack of row r does not wait on row r's final pack.
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
    constexpr uint32_t kLag = get_compile_time_arg_val(15);

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

    static_assert(num_k_blocks >= 2, "the last K-block is split off the helper's K-loop");
    constexpr auto shape = MatmulBlockShape::of(block_m / sb_h, n_tiles / sb_w, sb_h, sb_w, k_block, num_k_blocks - 1);
    // Last K-block runs one row at a time: widest divisor of n_tiles that fits the host's DEST subblock.
    constexpr uint32_t row_sb_w = [] {
        uint32_t w = sb_h * sb_w < n_tiles ? sb_h * sb_w : n_tiles;
        while (n_tiles % w) {
            --w;
        }
        return w;
    }();
    constexpr uint32_t act_block_tiles = block_m * k_block;
    constexpr uint32_t weight_block_tiles = k_block * n_tiles;

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

    // Last K-block of one row: accumulate onto the partials K-blocks 0..n-2 left in L1 (the row's partials
    // sit at the CB write pointer, which advances one row per push).
    auto last_k_row = [&](uint32_t row) {
        reconfig_data_format(cb_weight_block, cb_activation_block);
        ckernel::matmul_block_init(cb_activation_block, cb_weight_block, false, row_sb_w, 1, k_block);
        pack_reconfig_data_format(cb_matmul_partials);
        pack_reconfig_l1_acc(1);
        for (uint32_t col = 0; col < n_tiles; col += row_sb_w) {
            tile_regs_acquire();
            for (uint32_t inner = 0; inner < k_block; ++inner) {
                ckernel::matmul_block(
                    cb_activation_block,
                    cb_weight_block,
                    row * k_block + inner,
                    inner * n_tiles + col,
                    0,
                    false,
                    row_sb_w,
                    1,
                    k_block);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < row_sb_w; ++c) {
                pack_tile<true>(c, cb_matmul_partials, col + c);
            }
            tile_regs_release();
        }
        partials.push_back(n_tiles);
        pack_reconfig_l1_acc(0);
    };

    auto epilogue_setup = [&]() {
        if (is_rotary) {
            reconfig_data_format(cb_matmul_partials, cb_rotary_table);
            mul_init(cb_matmul_partials, cb_rotary_table, true);
        } else {
            reconfig_data_format_srca(cb_matmul_partials);
            copy_init(cb_matmul_partials);
        }
        pack_reconfig_data_format(cb_out_tiles);
    };

    auto epilogue_row_routed = [&](uint32_t row) {
        if (is_rotary) {
            cb_wait_front(cb_rotary_table, (row + 1 < seq_tiles ? row + 1 : seq_tiles) * table_tiles_per_row);
        }
        partials.wait_front((row + 1) * n_tiles);
        if constexpr (out_write_nocs == 2) {
            if (row & 1) {
                DataflowBuffer out_noc0_dfb(cb_out_tiles_noc0);
                epilogue_row(row, RotaryPackNoc0{out_noc0_dfb}, ConvertPackNoc0{out_noc0_dfb});
                return;
            }
        }
        epilogue_row(row, RotaryPack{out_dfb}, ConvertPack{out_dfb});
    };

    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;

        {
            MaybeDeviceZoneScope("compute_partials_reserve");
            partials.reserve_back(partials_tiles);
        }
        {
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

        // Last K-block interleaved with the epilogue: matmul row r + kLag, then the epilogue of row r.
        activation.wait_front(act_block_tiles);
        weight.wait_front(weight_block_tiles);
        MaybeDeviceZoneScope("compute_last_k_epilogue");
        const uint32_t lead = rows < kLag ? rows : kLag;
        for (uint32_t row = 0; row < lead; ++row) {
            last_k_row(row);
        }
        for (uint32_t row = 0; row < rows; ++row) {
            if (row + lead < rows) {
                last_k_row(row + lead);
            }
            epilogue_setup();
            epilogue_row_routed(row);
        }
        // Nominal rows past `rows` are never consumed; push them so the block's pop balances.
        if (rows < block_m) {
            partials.push_back((block_m - rows) * n_tiles);
        }
        activation.pop_front(act_block_tiles);
        weight.pop_front(weight_block_tiles);

        partials.wait_front(partials_tiles);
        partials.pop_front(partials_tiles);
        if (is_rotary) {
            const uint32_t table_tiles = (rows < seq_tiles ? rows : seq_tiles) * table_tiles_per_row;
            cb_pop_front(cb_rotary_table, table_tiles);
        }
    }
}
