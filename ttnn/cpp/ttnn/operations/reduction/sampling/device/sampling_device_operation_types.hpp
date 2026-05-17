// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct SamplingParams {
    std::optional<uint32_t> seed;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
};

struct SamplingInputs {
    Tensor input_values;
    Tensor input_indices;
    Tensor k;
    Tensor p;
    Tensor temp;
    std::optional<Tensor> preallocated_output;
    // tt-xla #4539 fix proposal: host-precomputed noise input.
    // Shape [32], bf16, ROW_MAJOR — same convention as k/p/temp. Required
    // on this branch — the kernel uses this instead of an internal RNG.
    // Stored as a plain Tensor (not std::optional) so the device-operation
    // framework's reflection tracks its buffer and updates the kernel's
    // runtime arg per launch.
    Tensor noise;
};

}  // namespace ttnn::prim
