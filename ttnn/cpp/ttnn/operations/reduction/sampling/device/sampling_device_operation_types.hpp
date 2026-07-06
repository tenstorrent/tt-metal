// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

struct SamplingParams {
    std::optional<uint32_t> seed;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    // WAR-hazard semaphore for safe reuse of a persistent all-gather buffer under trace.
    // When set, each sampling writer core increments `war_semaphore` (at the gather's drain core,
    // `war_sem_drain_core`) once at the very end of the op, signalling that this step's reads of the
    // SAMPLING_VALUES/INDICES buffers are complete. The next decode step's SAMPLING_VALUES all-gather
    // waits on its local copy at that core before overwriting the buffer, closing the cross-sub-device
    // Write-After-Read race that only manifests under trace. See models/common/sampling and the
    // llama3_70b_galaxy TT_CCL.
    std::optional<tt::tt_metal::GlobalSemaphore> war_semaphore;
    std::optional<tt::tt_metal::CoreCoord> war_sem_drain_core;
};

struct SamplingInputs {
    Tensor input_values;
    Tensor input_indices;
    Tensor k;
    Tensor p;
    Tensor temp;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
