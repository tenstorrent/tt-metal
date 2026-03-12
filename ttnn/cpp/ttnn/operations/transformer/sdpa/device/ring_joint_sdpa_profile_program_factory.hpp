// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_device_operation_types.hpp"

namespace ttnn::prim {

/**
 * Shared variables for profile program factory.
 * Simplified version without CCL all_gather shared variables.
 */
struct RingJointSDPAProfileSharedVariables {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreCoord grid_size;
    tt::tt_metal::KernelHandle reader_kernels_id{};
    tt::tt_metal::KernelHandle writer_kernels_id{};
    tt::tt_metal::KernelHandle compute_kernels_id{};
};

/**
 * Program factory for ring_joint_sdpa_profile.
 *
 * Key simplifications from RingJointSDPAProgramFactory:
 * - No CCL worker setup
 * - No fused_op_signaler
 * - ring_index passed as compile-time arg (not from device topology)
 * - Uses simplified reader/writer kernels
 */
struct RingJointSDPAProfileProgramFactory {
    using shared_variables_t = RingJointSDPAProfileSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RingJointSDPAProfileParams& args,
        const RingJointSDPAProfileInputs& tensor_args,
        RingJointSDPAProfileResult& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RingJointSDPAProfileParams& args,
        const RingJointSDPAProfileInputs& tensor_args,
        RingJointSDPAProfileResult& output_tensors);
};

}  // namespace ttnn::prim
