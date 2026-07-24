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
    // buffer-reuse sync semaphore for safe reuse of a persistent all-gather buffer under trace.
    std::optional<tt::tt_metal::GlobalSemaphore> buffer_reuse_sync_semaphore;
    std::optional<tt::tt_metal::CoreCoord> buffer_reuse_sync_sem_drain_core;
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
