// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"

namespace ttnn::prim::llama31_8b_matmul_bh::mcast_2d {

struct Mcast2DDefine {
    const char* key;
    const char* value;
};

struct Mcast2DRect {
    uint8_t x0, y0, x1, y1;
};

struct Mcast2DWorkUnit {
    const char* name;
    uint32_t kernel_mask;
    uint32_t first_rect, num_rects;
};

struct Mcast2DCore {
    uint8_t x, y, in0_kind, in1_kind;
    uint32_t in0_args[2];
    uint32_t out_args[12];
    uint32_t in1_args[15];
    uint32_t varargs[8];
};

struct Mcast2DPlan {
    uint32_t grid_x, grid_y, cores_c, cores_r;
    uint32_t Mt, Kt, Nt, in0_block_w, per_core_M, per_core_N;
    uint32_t out_block_h, out_block_w, out_subblock_h, out_subblock_w;
    uint32_t num_blocks, num_blocks_w_dim, num_blocks_h_dim;
    bool packer_l1_acc;
    uint32_t interm0_format;
    bool swiglu, out_writer_on_in0, share_out_interm;
    uint32_t in0_entries, in1_entries, out_entries, interm0_entries, swiglu_entries;
    uint32_t out_tile_bytes, in1_tile_bytes, num_dram_banks;
    bool has_in1_receiver, has_in0_receiver, has_other_noc;
    uint32_t in0_stride_h, in0_next_inner_dim_block_stride, in0_next_h_dim_block_stride, in0_block_num_tiles, MtKt;
    uint32_t in1_stride_h, in1_next_block_stride, in1_next_w_dim_block_stride, in1_block_num_tiles, KtNt;
    uint32_t out_stride_h, out_next_subblock_stride_h, out_next_h_dim_block_stride, out_subblock_tile_count, MtNt;
    uint32_t in1_dram_block_num_tiles, in1_block_w_dram_bytes, num_runtime_varargs;
    uint32_t in0_receiver_block_num_tiles, in0_num_subblocks, in0_subblock_num_tiles, in1_num_subblocks;
    uint32_t compute_in1_block_num_tiles, in1_per_core_w, out_subblock_num_tiles;
    uint32_t ow_stride_h, ow_next_subblock_stride_w, ow_next_subblock_stride_h, ow_next_w_dim_block_stride,
        ow_next_h_dim_block_stride;
    uint32_t ow_subblock_w, ow_subblock_tile_count, out_write_pace_cycles, out_write_pace_barrier;
    const Mcast2DDefine* defines_compute;
    uint32_t num_defines_compute;
    const Mcast2DDefine* defines_in0_send;
    uint32_t num_defines_in0_send;
    const Mcast2DDefine* defines_in0_recv;
    uint32_t num_defines_in0_recv;
    const Mcast2DDefine* defines_in1_send;
    uint32_t num_defines_in1_send;
    const Mcast2DDefine* defines_in1_recv;
    uint32_t num_defines_in1_recv;
    const Mcast2DDefine* defines_in1_recv_o;
    uint32_t num_defines_in1_recv_o;
    const Mcast2DRect* rects;
    const Mcast2DWorkUnit* units;
    uint32_t num_units;
    const Mcast2DCore* cores;
    uint32_t num_cores;
};

ttnn::device_operation::ProgramArtifacts create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const Mcast2DPlan& plan);

}  // namespace ttnn::prim::llama31_8b_matmul_bh::mcast_2d
