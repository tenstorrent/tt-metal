// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "convert_to_hwc_device_operation_types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

namespace ttnn::operations::experimental::cnn::program {

struct ConvertToHWCSharedVariables {
    tt::tt_metal::CBHandle cb_in{};
    tt::tt_metal::CBHandle cb_out{};
    bool is_input_in_dram = false;
    // Destination/output cores where kernels execute
    std::vector<tt::tt_metal::CoreCoord> output_cores;
    // Serialized per-core runtime args for gather-based writer kernels
    std::vector<std::vector<uint32_t>> per_core_serialized_transfers;
    tt::tt_metal::KernelHandle writer_kernel_id0{};
    tt::tt_metal::KernelHandle writer_kernel_id1{};
    uint32_t remote_address = 0;
};

struct ConvertToHWCProgramFactory {
    using shared_variables_t = ConvertToHWCSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::cnn::program

namespace ttnn::operations::experimental::cnn::detail {

// Named constants for circular buffer indices
namespace CBIndex {
constexpr uint32_t CB_IN = tt::CBIndex::c_0;
constexpr uint32_t CB_IN_BATCH = tt::CBIndex::c_1;
constexpr uint32_t CB_IN_TILED = tt::CBIndex::c_2;
constexpr uint32_t CB_IN_TRANSPOSE_0 = tt::CBIndex::c_3;
constexpr uint32_t CB_IN_TRANSPOSE_1 = tt::CBIndex::c_4;
constexpr uint32_t CB_OUT = tt::CBIndex::c_5;
}  // namespace CBIndex

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
    uint32_t remote_address{};
    tt::tt_metal::BufferType remote_buffer_type{};
    tt::CoreType remote_core_type{};

    // Alignment requirements
    uint32_t alignment_elements{};

    static ConvertToHwcConfig create_from_tensors(const Tensor& input, const Tensor& output);
    void validate() const;
};

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor);

}  // namespace ttnn::operations::experimental::cnn::detail
