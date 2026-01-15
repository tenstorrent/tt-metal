// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bcast_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::bcast::program {

// Shared variables for MULTI_CORE_H_SHARDED strategies
struct BcastShardedHSharedVariables {
    tt::tt_metal::KernelHandle binary_reader_kernel_id{};
    tt::tt_metal::KernelHandle bcast_kernel_id{};
    tt::tt_metal::CBHandle cb_src0{};
    tt::tt_metal::CBHandle out_cb{};
    uint32_t ncores_x{};
};

struct BcastShardedHProgramFactory {
    using shared_variables_t = BcastShardedHSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const BcastParams& operation_attributes,
        const BcastInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::bcast::program
