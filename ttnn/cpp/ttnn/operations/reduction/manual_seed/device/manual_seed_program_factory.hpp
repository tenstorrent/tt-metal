// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "manual_seed_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;

// Case 1: seed=uint32_t, user_ids=None - set all cores to the same seed
struct ManualSeedSingleSeedToAllCoresProgramFactory {
    struct shared_variables_t {};

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
    static void override_runtime_arguments(
        cached_program_t&, const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
};

// Case 2: seed=uint32_t, user_ids=uint32_t - set seed to one core based on user_id
struct ManualSeedSingleSeedSingleCoreProgramFactory {
    struct shared_variables_t {};

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
    static void override_runtime_arguments(
        cached_program_t&, const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
};

// Case 3: seed=uint32_t, user_ids=Tensor - set seeds to cores in user_ids tensor
struct ManualSeedSingleSeedSetCoresProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
    static void override_runtime_arguments(
        cached_program_t&, const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
};

// Case 4: seed=Tensor, user_ids=Tensor - set mapping seeds to cores based on tensors
struct ManualSeedSetSeedsSetCoresProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
    static void override_runtime_arguments(
        cached_program_t&, const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
};

}  // namespace ttnn::prim
