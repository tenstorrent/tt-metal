// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"

namespace ttnn::prim::llama31_8b_matmul_bh::mcast_1d {

struct Mcast1DRect {
    uint8_t x0, y0, x1, y1;
};

struct Mcast1DCore {
    uint8_t x, y;
    uint16_t in1_tensor_start_tile_id, out_tensor_start_tile_id, last_block_w;
    uint16_t out_last_num_nonzero_subblocks_w, out_last_subblock_w;
    uint32_t padded_subblock_tiles_addr_skip;
    uint16_t padded_block_tiles_w_skip, last_num_blocks_w_dim;
};

struct Mcast1DPlan {
    uint32_t grid_x, grid_y, Mt, Kt, Nt;
    uint32_t in0_block_w, per_core_M, per_core_N, out_subblock_h, out_subblock_w;
    uint32_t num_blocks, in0_group, num_cores, num_blocks_w_dim, num_blocks_h_dim;
    bool packer_l1_acc;
    uint32_t interm0_format;
    uint32_t in0_dfb_tiles, in1_dfb_tiles, out_dfb_tiles, interm0_dfb_tiles, out_tile_bytes;
    bool in1_coalesced_read, out_flush_per_block;
    uint32_t in0_stride_w, in0_stride_h, in0_next_inner_dim_block_stride, in0_next_h_dim_block_stride;
    uint32_t in0_reader_block_w, in0_block_h, in0_reader_block_num_tiles, in0_reader_num_blocks_inner, MtKt;
    uint32_t in1_stride_w, in1_stride_h, in1_next_block_stride, in1_next_w_dim_block_stride;
    uint32_t in1_block_w, in1_block_h, in1_block_num_tiles, KtNt;
    uint32_t out_block_num_tiles, out_stride_h, out_next_subblock_stride_w, out_next_subblock_stride_h;
    uint32_t out_next_w_dim_block_stride, out_next_h_dim_block_stride, out_subblock_tile_count, MtNt;
    uint32_t in0_num_subblocks, in0_block_num_tiles, in0_subblock_num_tiles;
    uint32_t in1_num_subblocks, in1_compute_block_num_tiles, in1_per_core_w, out_subblock_num_tiles;
    uint32_t in0_last_block_h, out_num_nonzero_subblocks_h, out_last_subblock_h, padded_block_tiles_h_skip;
    uint32_t out_num_nonzero_subblocks_w;
    Mcast1DRect bbox;
    const Mcast1DRect* all_cores;
    uint32_t num_all_cores;
    const Mcast1DRect* receivers;
    uint32_t num_receivers;
    const Mcast1DRect* tail;
    uint32_t num_tail;
    const Mcast1DCore* cores;
};

ttnn::device_operation::ProgramArtifacts create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const Mcast1DPlan& plan);

}  // namespace ttnn::prim::llama31_8b_matmul_bh::mcast_1d
