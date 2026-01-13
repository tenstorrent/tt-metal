// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_split_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::minimal_matmul::program {

struct MinimalMatmulSplitProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        std::vector<CoreCoord> cores;
        tt::tt_metal::KernelHandle in0_sender_kernels_id{};
        tt::tt_metal::KernelHandle in0_receiver_kernels_id{};
        tt::tt_metal::KernelHandle in1_sender_kernels_id{};
        tt::tt_metal::KernelHandle in1_receiver_kernels_id{};
        bool transpose_core_grid{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const split_operation_attributes_t& operation_attributes,
        const split_tensor_args_t& tensor_args,
        split_tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const split_operation_attributes_t& operation_attributes,
        const split_tensor_args_t& tensor_args,
        split_tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::minimal_matmul::program
