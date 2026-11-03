#define PT_NORECONFIG 1
#define PT_BATCH 1
#define PT_NOSTALL 1
// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// linear_gelu compute: project_gelu_block = one matmul_block call per (row-block, n-block):
// in0 = [x | ones] (rows x k_ext_tiles, retained across the n-blocks), in1 = W chunk j of the resident slice
// ([W ; bias] k_ext_tiles x block_w, caller-managed), one K-block accumulated in fp32 DEST, tanh GELU applied
// on the packer per subblock (raw-LLK load-macro body, see linear_gelu_gelu_sfpu.h), packed subblock-major to
// cb_output_tiles.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "mbh_local.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "linear_gelu_gelu_sfpu.h"

namespace {

struct WeightChunkOffset {
    uint32_t base;
    ALWI uint32_t operator()(uint32_t /*block*/) const { return base; }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_activation_block = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_resident = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr uint32_t block_w = get_compile_time_arg_val(4);
    constexpr uint32_t k_ext_tiles = get_compile_time_arg_val(5);

    const uint32_t core_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_n_blocks = get_arg_val<uint32_t>(1);
    const uint32_t sb_h = get_arg_val<uint32_t>(2);
    const uint32_t sb_w = get_arg_val<uint32_t>(3);
    const uint32_t tail_sb_h = get_arg_val<uint32_t>(4);
    const uint32_t tail_sb_w = get_arg_val<uint32_t>(5);

    static_assert(DST_ACCUM_MODE, "the load-macro GELU stores fp32 DEST unrounded");
    using Gelu = GeluTanhActivation<kGeluLoadMacro>;
    constexpr uint32_t act_block_tiles = block_h * k_ext_tiles;
    constexpr uint32_t w_chunk_tiles = k_ext_tiles * block_w;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_activation_block, cb_weight_resident, cb_output_tiles);
    ActivationInitHelper<Gelu::activation, Gelu::param0>::init();

    CircularBuffer act_buf(cb_activation_block);
    CircularBuffer weight_buf(cb_weight_resident);
    CircularBuffer out_buf(cb_output_tiles);

    {
        // Fill: unpack waits for x row-block 0 and W chunk 0 (the helper's own waits then pass at once).
        MaybeDeviceZoneScope("compute_fill");
        act_buf.wait_front(act_block_tiles);
        weight_buf.wait_front(w_chunk_tiles);
    }

    const uint32_t num_row_blocks = (core_rows + block_h - 1) / block_h;
    // matmul_block_init carries the subblock dims, so re-init only when they change (nominal -> tail).
    uint32_t init_sb_h = 0;
    uint32_t init_sb_w = 0;
    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        if (row_block_idx > 0) {
            // Starvation on x row-block i (row-block 0's wait is compute_fill); the helper's own wait then passes.
            MaybeDeviceZoneScope("compute_wait_x");
            act_buf.wait_front(act_block_tiles);
        }
        // Occupancy: matmul + packer GELU for one row-block, including the in-helper W / output CB waits.
        MaybeDeviceZoneScope("compute_matmul_gelu");
        const uint32_t first_row = row_block_idx * block_h;
        const bool tail = core_rows - first_row < block_h;
        const uint32_t rows = tail ? core_rows - first_row : block_h;
        const uint32_t cur_sb_h = tail ? tail_sb_h : sb_h;
        const uint32_t cur_sb_w = tail ? tail_sb_w : sb_w;
        const auto shape =
            MatmulBlockShape::of(rows / cur_sb_h, block_w / cur_sb_w, cur_sb_h, cur_sb_w, k_ext_tiles, 1);

        for (uint32_t n_block_idx = 0; n_block_idx < num_n_blocks; ++n_block_idx) {
            // in1 is NoWaitNoPop: cumulative wait from the never-moved read pointer (free after row-block 0).
            weight_buf.wait_front((n_block_idx + 1) * w_chunk_tiles);
            const WeightChunkOffset chunk_offset{n_block_idx * w_chunk_tiles};
            if (cur_sb_h != init_sb_h || cur_sb_w != init_sb_w) {
                init_sb_h = cur_sb_h;
                init_sb_w = cur_sb_w;
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::NoWaitNoPop,
                    NoPostCompute,
                    NoPreKBlock,
                    NoPostKBlock,
                    0,
                    NoKBlockInnerDimFn,
                    NoIn0Source,
                    WeightChunkOffset,
                    false,
                    false,
                    Gelu,
                    matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                    act_buf, weight_buf, out_buf, out_buf, shape, {}, {}, block_w, 0, {}, {}, {}, chunk_offset);
            } else {
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::None,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::NoWaitNoPop,
                    NoPostCompute,
                    NoPreKBlock,
                    NoPostKBlock,
                    0,
                    NoKBlockInnerDimFn,
                    NoIn0Source,
                    WeightChunkOffset,
                    false,
                    false,
                    Gelu,
                    matmul_config::DataFormatReconfig::NONE>(
                    act_buf, weight_buf, out_buf, out_buf, shape, {}, {}, block_w, 0, {}, {}, {}, chunk_offset);
            }
        }
        // release_activation_block: nominal quantum, matching the reader's push on the tail.
        act_buf.pop_front(act_block_tiles);
    }
}
