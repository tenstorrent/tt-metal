// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/kernel_types.hpp"

namespace ttnn::prim {

struct TilizeParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    bool use_multicore = false;
    bool enough_space_width = false;
    bool enough_space_height = false;
    const bool use_low_perf = false;
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

struct TilizeInputs {
    Tensor input_tensor;
    std::optional<Tensor> optional_input_tensor;
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
