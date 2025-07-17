// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "conv2d/device/conv2d_op.hpp"

#include "tt-metalium/circular_buffer_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::conv {
namespace conv2d {

// Invalid value for cb id is 32, number greater than the maximum number of index circular buffer can have.
constexpr static uint32_t kInvalidCBIndex = 32;

// List of all circular buffers used in Conv2d operations.
enum class Conv2dCb {
    ACT_SHARDED,
    ACT,
    ACT_ROW_MAJOR_BFLOAT16,
    ACT_SECOND_READER,
    ACT_TILIZED,
    WEIGHTS,
    BIAS,
    READER_INDICES,
    L1_ARRAY,
    MATMUL_PARTIALS,
    OUT,
    TEMP_SUM,
    COUNT
};
struct CBInfo {
    // Index of CB that will be passed in to the kernel.
    uint32_t index = kInvalidCBIndex;
    // CB handle
    tt::tt_metal::CBHandle handle;
    // Type of the CB
    Conv2dCb name;
    // Number of pages in the circular buffer.
    uint32_t num_pages;
    // Size of each page in the circular buffer.
    uint32_t page_size;
    // Whether this CB is globally allocated (true for sharded tensors).
    bool is_globally_allocated = false;
    // Data format of the circular buffer.
    tt::DataFormat data_format = tt::DataFormat::Invalid;
    // Optional: If this CB is overlapped by another CB, this will hold the name of that CB.
    std::optional<Conv2dCb> overlapped_by_cb = std::nullopt;

    uint32_t cb_size_per_core() const { return num_pages * page_size; }
};

// Returns a vector of CBInfo objects for the Conv2d operation.
// The vector will contain information about all circular buffers used in the Conv2d operation.
// CBInfo::index and CBInfo::handle won't be valid until allocate_cbs() is called.
std::vector<CBInfo> get_cb_info(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const OptimizedConvBlockConfig& block_config,
    const OptimizedConvParallelizationConfig& pconfig,
    const ttnn::Shape& weights_shape,
    std::array<uint32_t, 2> kernel_size,
    const Conv2dConfig& conv_config,
    DataType input_datatype,
    DataType output_datatype,
    std::array<uint32_t, 2> conv_input_shard_shape,
    bool enable_bias,
    bool is_1d_depthwise_conv,
    bool skip_act_cb_create);

// Allocates circular buffers for the Conv2d operation.
// This function will populate index and handle fields of each CBInfo in the cb_info vector,
// and add these circular buffers to the provided program.
void allocate_cbs(
    std::vector<CBInfo>& cb_info,
    tt::tt_metal::Program& program,
    const CoreRange& all_cores,
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& l1_indices_tensor);

const CBInfo& get_cb_info_by_name(const std::vector<CBInfo>& cb_info, Conv2dCb cb_name);
CBInfo& access_cb_info_by_name(const std::vector<CBInfo>& cb_info, Conv2dCb cb_name);

}  // namespace conv2d
}  // namespace ttnn::operations::conv
