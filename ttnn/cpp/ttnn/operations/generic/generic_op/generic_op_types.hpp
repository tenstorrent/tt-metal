// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include "ttnn/decorators.hpp"

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
namespace ttnn::operations::generic {

struct circular_buffer_attributes_t {
    CoreRangeSet core_spec;
    uint32_t total_size;
    uint32_t page_size;
    // uint8_t buffer_index;
    tt::DataFormat data_format;

    // this needs better solution as we now have input tensors (std::vector) and output tensor so index is not great
    std::optional<int> set_globally_allocated_address =
        std::nullopt;  // an index to io_tensors that will set globally allocated address on CB
};

using cb_attr_map = std::unordered_map<tt::CBIndex, circular_buffer_attributes_t>;

struct data_movement_attributes_t {
    CoreRangeSet core_spec;
    std::string kernel_path;
    tt::tt_metal::DataMovementConfig config;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core = {};

    // std::variant<CoreCoord, CoreRange, CoreRangeSet> core_spec;
    // std::shared_ptr<RuntimeArgs> runtime_args;
    // std::vector<std::shared_ptr<RuntimeArgs>> runtime_args;
};

struct compute_attributes_t {
    CoreRangeSet core_spec;
    std::string kernel_path;
    tt::tt_metal::ComputeConfig config;
    // std::vector<uint32_t> runtime_args = {};
    std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core = {};
};

struct program_attributes_t {
    cb_attr_map circular_buffer_attributes;
    std::vector<data_movement_attributes_t> data_movement_attributes;
    std::vector<compute_attributes_t> compute_attributes;
};

}  // namespace ttnn::operations::generic
