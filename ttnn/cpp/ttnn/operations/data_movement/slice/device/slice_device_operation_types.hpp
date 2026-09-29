// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct SliceParams {
    ttnn::Shape slice_start;
    ttnn::Shape slice_end;
    ttnn::Shape step;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_tensor_args = false;
    std::optional<uint32_t> slice_dim = std::nullopt;
    std::optional<uint32_t> num_devices = std::nullopt;
    std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
    // True when output_mem_config was defaulted from the input (no caller-supplied memory_config /
    // preallocated output), so compute_output_specs may rescale an inherited ND shard shape to the
    // sliced output. Explicit configs must be honored verbatim.
    // Only the tensor-args overload sets this. The span overload has already rescaled in slice.cpp and
    // leaves it false: rescaling again against the input's padded shape is not idempotent.
    bool output_mem_config_inherited = false;
};

struct SliceInputs {
    Tensor input;
    std::optional<Tensor> start_tensor;
    std::optional<Tensor> end_tensor;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
