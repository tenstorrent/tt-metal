// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "conv3d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

// The circular-buffer sizes a conv3d program needs however much L1 is free, derived from the attributes and
// the tensor specs alone. plan_l1 decides the L1-dependent parts of the program from these.
struct Conv3dCbGeometry {
    uint32_t matmul_M_t = 0;
    uint32_t matmul_K_t = 0;
    uint32_t matmul_N_t = 0;
    uint32_t C_in_num_blocks = 0;  // more than one: the reduction and worker_ack CBs
    uint32_t tile_size = 0;
    uint32_t partial_tile_size = 0;
    bool use_fp32_exact = false;
    bool use_fp32_partials = false;  // with use_fp32_exact: the running-partial CB
    bool use_bias = false;
    bool operand_split = false;  // the x_hi, x_lo and weight_lo CBs
    uint32_t vol2col_rm_pages = 0;
    uint32_t padded_patch_size_bytes = 0;
    uint32_t C_in_block_bytes = 0;
    uint32_t out_subblock_h = 1;  // before the L1 shrink
    // The reader's DRAM read staging and the logical-pad offset page.
    uint32_t in_row_size_bytes = 0;
    bool input_is_dram_interleaved = false;
    uint32_t dram_read_alignment = 0;
    bool enable_dram_read_staging = false;
    uint32_t dram_read_scratch_page_bytes = 0;  // 0 without staging
    bool mask_mode = false;
    uint32_t pad_offset_page_bytes = 0;  // 0 without masking
    // The prefetch shard holds one spatial block's receptive field: shard_positions rows of C_in_block_bytes.
    // shard_positions is 0 when the kernel has no spatial reuse or is dilated, so there is no prefetch.
    uint32_t T_shard = 0;
    uint32_t H_shard = 0;
    uint32_t W_shard = 0;
    uint32_t shard_positions = 0;
    uint32_t shard_rows = 0;  // T_shard * H_shard; one coalesced-read scratch row holds W_shard positions
    bool coalesced_candidate = false;
    uint32_t num_dram_banks = 0;
};

// The decisions that depend on how much L1 the circular buffers may take: out_subblock_h after the Wormhole
// shrink, whether the input shard CB exists, and how many rows of coalesced-read scratch it carries. The
// program cache keys on these, not on the budget they were decided from, so a move of the L1 frontier
// re-plans only when it changes a decision.
struct Conv3dL1Plan {
    uint32_t out_subblock_h = 1;
    bool prefetch_shard = false;
    uint32_t coalesced_scratch_rows = 0;
    uint32_t l1_prefetch_max_bytes = 0;  // what the shard was decided against; informational, not keyed on
};

Conv3dCbGeometry derive_cb_geometry(const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args);
Conv3dL1Plan plan_l1(const Conv3dCbGeometry& geometry, uint32_t l1_usable_for_cbs);
// The plan against the live budget: the L1 span from the allocator's base up to the lowest live L1 buffer
// (available_program_l1_capacity). The op's own output is allocated before either caller runs, so the
// program hash and the program factory see the same budget.
Conv3dL1Plan select_l1_plan(const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args);

struct Conv3dProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
