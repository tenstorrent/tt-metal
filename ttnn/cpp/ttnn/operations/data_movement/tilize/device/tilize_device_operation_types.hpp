// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <tuple>

namespace ttnn::prim {

struct TilizeParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    bool use_multicore = false;
    bool enough_space_width = false;
    bool enough_space_height = false;
    const bool use_low_perf = false;
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "output_mem_config",
        "output_dtype",
        "use_multicore",
        "enough_space_width",
        "enough_space_height",
        "use_low_perf",
        "sub_core_grids");
    auto attribute_values() const {
        return std::forward_as_tuple(
            output_mem_config,
            output_dtype,
            use_multicore,
            enough_space_width,
            enough_space_height,
            use_low_perf,
            sub_core_grids);
    }
};

struct TilizeInputs {
    Tensor input_tensor;
    std::optional<Tensor> optional_input_tensor;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "optional_input_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, optional_input_tensor); }
};

struct MultiCoreSharedVariables {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
        std::vector<CoreCoord> cores;
        uint32_t ncores{};
    };
};

}  // namespace ttnn::prim
