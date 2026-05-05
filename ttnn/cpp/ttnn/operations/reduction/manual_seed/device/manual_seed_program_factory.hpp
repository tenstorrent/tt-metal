// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "manual_seed_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;

// Case 1: seed=uint32_t, user_ids=None - set all cores to the same seed
struct ManualSeedSingleSeedToAllCoresProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
};

// Case 2: seed=uint32_t, user_ids=uint32_t - set seed to one core based on user_id
struct ManualSeedSingleSeedSingleCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
};

// Case 3: seed=uint32_t, user_ids=Tensor - set seeds to cores in user_ids tensor
struct ManualSeedSingleSeedSetCoresProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
};

// Case 4: seed=Tensor, user_ids=Tensor - set mapping seeds to cores based on tensors
struct ManualSeedSetSeedsSetCoresProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(const ManualSeedParams&, const ManualSeedInputs&, Tensor&);
};

}  // namespace ttnn::prim
