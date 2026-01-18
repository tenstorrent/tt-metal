// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "group_attn_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GroupAttnMatmulSharedVariables {
    tt::tt_metal::KernelHandle reader_id = 0;
    tt::tt_metal::KernelHandle writer_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::CBHandle cb_src0 = 0;
    tt::tt_metal::CBHandle cb_src1 = 0;
    tt::tt_metal::CBHandle cb_src2 = 0;  // 0 if unused
    tt::tt_metal::CBHandle cb_interm1 = 0;
    tt::tt_metal::CBHandle cb_output = 0;
    uint32_t in1_mcast_sender_semaphore_id = 0;
    uint32_t in1_mcast_receiver_semaphore_id = 0;
    std::vector<uint32_t> in1_mcast_sender_noc_x;
    std::vector<uint32_t> in1_mcast_sender_noc_y;
    uint32_t in0_single_tile_size = 0;
    uint32_t in1_single_tile_size = 0;
    uint32_t interm_single_tile_size = 0;
    uint32_t output_single_tile_size = 0;
    bool in0_is_sharded = false;
    bool in1_is_sharded = false;
    bool output_is_sharded = false;
    bool reader_noc_is_NOC_0 = false;
    uint32_t out_subblock_w = 0;
    uint32_t out_subblock_h = 0;
    uint32_t in1_num_subblocks = 0;
    uint32_t out_block_w = 0;
    uint32_t in1_per_core_w = 0;
    uint32_t in1_block_w_tile_bytes = 0;
    uint32_t ONE_ROW_BFLOAT16_BYTES = 0;
    uint32_t bfloat16_row_bytes = 0;
    CoreCoord device_compute_with_storage_grid;
};

struct GroupAttnMatmulProgramFactory {
    using shared_variables_t = GroupAttnMatmulSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GroupAttnMatmulParams& operation_attributes,
        const GroupAttnMatmulInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GroupAttnMatmulParams& operation_attributes,
        const GroupAttnMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
