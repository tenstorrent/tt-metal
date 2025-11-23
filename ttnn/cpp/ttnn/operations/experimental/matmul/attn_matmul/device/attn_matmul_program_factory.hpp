// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "attn_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::matmul::attn_matmul::program {

struct AttnMatmulProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_id{};
        tt::tt_metal::KernelHandle writer_id{};
        tt::tt_metal::KernelHandle eltwise_binary_kernel_id{};
        uint32_t total_num_cores = 0;
        uint32_t in0_single_tile_size = 0;
        tt::tt_metal::CBHandle cb_src0{};
        uint32_t src0_cb_index = 0;
        uint32_t num_cores_y = 0;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::matmul::attn_matmul::program
