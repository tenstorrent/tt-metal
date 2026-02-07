// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

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

    static constexpr auto attribute_names = std::forward_as_tuple(
        "slice_start",
        "slice_end",
        "step",
        "output_mem_config",
        "use_tensor_args",
        "slice_dim",
        "num_devices",
        "sub_core_grids");
    auto attribute_values() const {
        return std::forward_as_tuple(
            slice_start, slice_end, step, output_mem_config, use_tensor_args, slice_dim, num_devices, sub_core_grids);
    }
};

struct SliceInputs {
    Tensor input;
    std::optional<Tensor> start_tensor;
    std::optional<Tensor> end_tensor;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input", "start_tensor", "end_tensor", "preallocated_output");
    auto attribute_values() const {
        return std::forward_as_tuple(input, start_tensor, end_tensor, preallocated_output);
    }
};

}  // namespace ttnn::prim
