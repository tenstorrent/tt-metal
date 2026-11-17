// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// genify_matmul compute — block operations `matmul_accumulate_block` + `epilogue_block` of regime R1.
//
// Epilogue none: one compute_kernel_lib::matmul_block call per block of this core's region. The
// helper walks the num_k_blocks K-blocks (waiting / popping cb_act_tiles and cb_weight_tiles per
// K-block), accumulates the partial sums of the whole block in cb_partials with packer L1
// accumulation (TileRowMajor spill, exact one-block capacity), reloads them on the last K-block and
// packs the result row group by row group into cb_output_tiles. With a single K-block there is no
// accumulator: cb_output_tiles is passed as the helper's same-format interm placeholder.
// The ragged last K-block is handled by LastKBlockInnerDim (only last_block_k_tiles FMA steps).
//
// Epilogue != none (op_design.md Block Operation Realization 2', 3a-3c): matmul_block packs its last
// K-block into cb_partials (LastBlockTarget::Interm, the whole block stays fronted there), then ONE
// epilogue block op reads it and packs cb_output_tiles row by row:
//   bias / bias_relu  add_bias_bcast_rows<RowBroadcast, TileRowMajor> (+ PACK_RELU for bias_relu)
//   swiglu            eltwise_chain over (block_m, block_n/2): gate (Strided base 0) -> Silu,
//                     up (Strided base block_n/2), MulBinary; the reader permuted B so all gate tiles
//                     of the N-block precede all up tiles.
//   addcmul           eltwise_chain over (block_m, block_n): FPU Mul with the row-broadcast gate, then
//                     SFPU AddBinary of the streamed residual (no DEST reuse ahead of an FPU op).
// The bias / gate row is waited once before the first batch's epilogue and popped after the last.
//
// Silu: the chain library has no Silu element, so a local one is declared below through the chain's
// documented UnaryOp CRTP extension point (silu_tile / silu_tile_init). It replaces the design's
// Sigmoid + MulBinary composition (one SFPU pass and one DEST copy fewer per output tile); it is a
// chain element, not a raw-LLK bypass of the chain (the chain still owns init, DEST sync, CB sync and
// data-format reconfig).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"  // MulBinary, AddBinary

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_weight_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_partials = get_compile_time_arg_val(3);
constexpr uint32_t block_batches = get_compile_time_arg_val(4);
constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(5);
constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(6);
constexpr uint32_t sb_h = get_compile_time_arg_val(7);
constexpr uint32_t sb_w = get_compile_time_arg_val(8);
constexpr uint32_t block_k_tiles = get_compile_time_arg_val(9);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(10);
constexpr uint32_t last_block_k_tiles = get_compile_time_arg_val(11);
constexpr uint32_t block_n_tiles = get_compile_time_arg_val(12);
constexpr uint32_t epilogue = get_compile_time_arg_val(13);  // 0 none, 1 bias, 2 bias_relu, 3 swiglu, 4 addcmul
constexpr uint32_t cb_row = get_compile_time_arg_val(14);    // bias (1, 2) or gate (4) N-block row
constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(15);
constexpr uint32_t block_m_tiles = get_compile_time_arg_val(16);
constexpr uint32_t epilogue_block_tiles = get_compile_time_arg_val(17);  // chain DEST lanes per acquire
// 1: silu runs the full-precision (fp32) sigmoid; 0: the bf16-precision sigmoid (host picks 0 only
// for a non-fp32 output, whose own rounding dominates).
constexpr bool silu_fp32_precision = get_compile_time_arg_val(18) != 0;

constexpr bool has_epilogue = epilogue != 0;
constexpr bool is_bias = epilogue == 1 || epilogue == 2;
constexpr bool is_bias_relu = epilogue == 2;
constexpr bool is_swiglu = epilogue == 3;
constexpr bool is_addcmul = epilogue == 4;
constexpr bool has_row = is_bias || is_addcmul;
constexpr uint32_t half_n = block_n_tiles / 2;
constexpr uint32_t block_tiles = block_m_tiles * block_n_tiles;

// FMA step count per K-block: the balanced split leaves the last K-block narrower; the reader
// pushes the nominal block_k pages, so the garbage tail must never be summed.
struct LastKBlockInnerDim {
    ALWI uint32_t operator()(uint32_t block, uint32_t block_k) const {
        return (block == num_k_blocks - 1) ? last_block_k_tiles : block_k;
    }
};

// silu(x) = x * sigmoid(x) in one SFPU pass (see the file header). `fp32_precision` selects the
// sigmoid algorithm (silu_tile's template arg); DEST is read and written in its configured width either way.
template <bool fp32_precision, ckl::Dst Slot = ckl::Dst::D0>
struct Silu : ckl::UnaryOp<Silu<fp32_precision, Slot>, Slot> {
    static ALWI void init() { silu_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        silu_tile<fp32_precision && DST_ACCUM_MODE>(ckl::to_u32(Slot) + slot_offset);
    }
};

// Epilogue none: no accumulator CB when there is a single K-block; the helper only needs a
// same-format placeholder. Any epilogue always lands the last K-block in cb_partials.
constexpr uint32_t cb_interm = (has_epilogue || num_k_blocks > 1) ? cb_partials : cb_output_tiles;

template <ckl::LastBlockTarget target>
ALWI void run_matmul(
    CircularBuffer& act_buf,
    CircularBuffer& weight_buf,
    CircularBuffer& out_buf,
    CircularBuffer& interm_buf,
    const ckl::MatmulBlockShape& shape) {
    ckl::matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/true,
        target,
        ckl::OutputCBLayout::TileRowMajor,
        ckl::matmul_config::InitMode::Short,
        ckl::InputPolicy::WaitAndPopPerKBlock,
        ckl::InputPolicy::WaitAndPopPerKBlock,
        ckl::NoPostCompute,
        ckl::NoPreKBlock,
        ckl::NoPostKBlock,
        /*untilize_block_ct_dim=*/0,
        LastKBlockInnerDim>(
        act_buf,
        weight_buf,
        out_buf,
        interm_buf,
        shape,
        ckl::NoPostCompute{},
        ckl::NoPreKBlock{},
        /*in1_per_core_w=*/block_n_tiles,
        /*out_row_width=*/block_n_tiles,
        ckl::NoPostKBlock{},
        LastKBlockInnerDim{});
}

// epilogue_block: cb_partials (one fronted block) -> cb_output_tiles (row by row).
ALWI void run_epilogue(
    CircularBuffer& partials_buf,
    CircularBuffer& row_buf,
    CircularBuffer& out_buf,
    DataflowBuffer& partials_dfb,
    DataflowBuffer& row_dfb,
    DataflowBuffer& residual_dfb,
    DataflowBuffer& out_dfb) {
    using namespace ckl;
    if constexpr (is_bias) {
        // The helper waits / pops cb_partials per row group; the bias row lifecycle is the caller's.
        if constexpr (is_bias_relu) {
            pack_relu_config(ReluConfig::zero());
        }
        add_bias_bcast_rows<BiasBroadcast::RowBroadcast, OutputCBLayout::TileRowMajor>(
            partials_buf,
            row_buf,
            out_buf,
            BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, sb_h, sb_w, block_n_tiles));
        if constexpr (is_bias_relu) {
            pack_relu_config(ReluConfig::none());
        }
    } else if constexpr (is_swiglu) {
        // Strided addressing requires caller-managed (None, None) lifecycles on cb_partials.
        partials_buf.wait_front(block_tiles);
        eltwise_chain(
            IterationShape::grid(block_m_tiles, half_n).block_size(epilogue_block_tiles),
            CopyTile<
                input(
                    cb_partials,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Strided),
                Dst::D0>{partials_dfb, StridedTileRange{0, block_n_tiles}},
            Silu<silu_fp32_precision, Dst::D0>{},
            CopyTile<
                input(
                    cb_partials,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Strided),
                Dst::D1>{partials_dfb, StridedTileRange{half_n, block_n_tiles}},
            MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>{out_dfb});
        partials_buf.pop_front(block_tiles);
    } else if constexpr (is_addcmul) {
        partials_buf.wait_front(block_tiles);
        eltwise_chain(
            IterationShape::grid(block_m_tiles, block_n_tiles),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(cb_partials, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
                input(cb_row, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row),
                Dst::D0>{partials_dfb, row_dfb},
            CopyTile<input(cb_residual_tiles), Dst::D1>{residual_dfb},
            AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>{out_dfb});
        partials_buf.pop_front(block_tiles);
    }
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // matmul maps in0 -> SrcB, in1 -> SrcA: Reverse is mandatory.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_tiles, cb_weight_tiles, cb_output_tiles);

    CircularBuffer act_buf(cb_act_tiles), weight_buf(cb_weight_tiles), out_buf(cb_output_tiles), interm_buf(cb_interm);

    if constexpr (!has_epilogue) {
        const auto shape = ckl::MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, sb_h, sb_w, block_k_tiles, num_k_blocks, block_batches);
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            run_matmul<ckl::LastBlockTarget::Out>(act_buf, weight_buf, out_buf, interm_buf, shape);
        }
    } else {
        // One batch slice per matmul call: cb_partials holds exactly one block, and the epilogue
        // must drain it before the next slice is accumulated.
        const auto shape = ckl::MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, sb_h, sb_w, block_k_tiles, num_k_blocks, /*batch=*/1);
        CircularBuffer row_buf(has_row ? cb_row : cb_partials);
        DataflowBuffer partials_dfb(cb_partials), row_dfb(has_row ? cb_row : cb_partials),
            residual_dfb(is_addcmul ? cb_residual_tiles : cb_partials), out_dfb(cb_output_tiles);
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            for (uint32_t bi = 0; bi < block_batches; ++bi) {
                run_matmul<ckl::LastBlockTarget::Interm>(act_buf, weight_buf, out_buf, interm_buf, shape);
                // The helper leaves packer L1 accumulation on after a multi-K-block walk; the
                // epilogue packs fresh tiles (bias helper prerequisite: caller disables L1_ACC).
                pack_reconfig_l1_acc(0);
                if constexpr (has_row) {
                    if (bi == 0) {
                        row_buf.wait_front(block_n_tiles);
                    }
                }
                run_epilogue(interm_buf, row_buf, out_buf, partials_dfb, row_dfb, residual_dfb, out_dfb);
            }
            if constexpr (has_row) {
                row_buf.pop_front(block_n_tiles);
            }
        }
    }
}
