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
// shrink, whether the input shard CB exists, and how many rows of coalesced-read scratch it carries.
struct Conv3dL1Plan {
    uint32_t out_subblock_h = 1;
    bool prefetch_shard = false;
    uint32_t coalesced_scratch_rows = 0;
    uint32_t static_cb_bytes = 0;        // every CB but the prefetch shard, at out_subblock_h
    uint32_t l1_prefetch_max_bytes = 0;  // what the shard was decided against; informational
};

// The L1 the circular buffers may take on this device. span: everything above the allocator's L1 base, the
// bound the enqueue-time region check applies when no L1 buffer is live. planning: the span less
// L1_TENSOR_HEADROOM, kept free for the L1 buffers live when the program runs; the plan shrinks and prefetches
// against it. Both follow from the device configuration only, so the plan is as static as the program cache
// key and as a traced program.
inline constexpr uint32_t L1_TENSOR_HEADROOM = 128 * 1024;
struct Conv3dL1Budget {
    uint32_t span = 0;
    uint32_t planning = 0;
};

Conv3dCbGeometry derive_cb_geometry(const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args);
Conv3dL1Plan plan_l1(const Conv3dCbGeometry& geometry, const Conv3dL1Budget& budget);
Conv3dL1Budget conv3d_l1_cb_budget(const tt::tt_metal::IDevice* device);
// derive_cb_geometry and plan_l1 against conv3d_l1_cb_budget: what validate_on_program_cache_miss checks and
// what create_descriptor builds.
Conv3dL1Plan select_l1_plan(const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args);

struct Conv3dProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
