// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation_types.hpp"

namespace ttnn::prim {

struct JointSDPASharedVariables {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreCoord grid_size;
    tt::tt_metal::KernelHandle reader_kernels_id{};
    tt::tt_metal::KernelHandle writer_kernels_id{};
    tt::tt_metal::KernelHandle compute_kernels_id{};
};

struct JointSDPAProgramFactory {
    using shared_variables_t = JointSDPASharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const JointSDPAParams& args, const JointSDPAInputs& tensor_args, JointSDPAResult& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const JointSDPAParams& args,
        const JointSDPAInputs& tensor_args,
        JointSDPAResult& output_tensors);
};

}  // namespace ttnn::prim
