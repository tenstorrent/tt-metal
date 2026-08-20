// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "convert_to_hwc_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

namespace ttnn::experimental::prim {

struct ConvertToHWCProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConvertToHwcParams& operation_attributes,
        const ConvertToHwcInputs& tensor_args,
        Tensor& tensor_return_value);
};

// Configuration class to encapsulate operation parameters
struct ConvertToHwcConfig {
    // Input tensor properties
    uint32_t batch_size{};
    uint32_t input_channels{};
    uint32_t hw_total{};
    uint32_t element_size_bytes{};
    tt::DataFormat input_format{};

    // Shard specifications
    uint32_t l1_input_shard_height{};
    uint32_t l1_input_shard_width{};
    uint32_t output_shard_height{};
    uint32_t output_shard_width{};

    // Gather output shard specifications for CB_IN_BATCH and transfer calculations
    uint32_t gather_l1_output_shard_height{};
    uint32_t gather_l1_output_shard_width{};

    // Core information
    std::vector<CoreCoord> l1_input_cores;
    std::vector<CoreCoord> dram_input_cores;
    CoreRangeSet l1_input_core_grid;
    std::vector<CoreCoord> output_cores;
    CoreRangeSet output_core_grid;

    // DRAM/L1 configuration
    bool is_input_in_dram{};
    tt::tt_metal::BufferType remote_buffer_type{};

    // Alignment requirements
    uint32_t alignment_elements{};

    static ConvertToHwcConfig create_from_tensors(const Tensor& input, const Tensor& output);
    void validate() const;
};

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor);

}  // namespace ttnn::experimental::prim
