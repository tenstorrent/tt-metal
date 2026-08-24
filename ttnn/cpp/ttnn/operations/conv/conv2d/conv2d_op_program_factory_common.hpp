// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp"
#include "tt-metalium/experimental/metal2_host_api/program_spec.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Preserve the legacy Conv2D ProgramDescriptor contract while lowering to Metalium 2.0: only math
// fidelity and FP32 destination accumulation were forwarded; all other hardware knobs kept defaults.
tt::tt_metal::experimental::ComputeHardwareConfig make_legacy_conv2d_compute_hardware_config(
    const ComputeKernelConfig& compute_kernel_config);

// Complete a legacy Gen1 CB topology using the supported fictional-endpoint recipe. Every kernel
// already bound to the DFB receives the opposite role under the same accessor name. Bindings do not
// emit synchronization instructions; the existing Gen1 plain-CB multi-binding path preserves the
// physical CB and kernel-owned handshake on both active and no-op placement nodes.
void add_fictional_dfb_endpoints(
    tt::tt_metal::experimental::ProgramSpec& spec, const tt::tt_metal::experimental::DFBSpecName& dfb_name);

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
    MATMUL_PARTIALS,
    OUT,
    COUNT
};
struct CBInfo {
    // Type of the CB
    Conv2dCb name{Conv2dCb::COUNT};
    // Number of pages in the circular buffer.
    uint32_t num_pages{};
    // Size of each page in the circular buffer.
    uint32_t page_size{};
    // Whether this CB is globally allocated (true for sharded tensors).
    bool is_globally_allocated = false;
    // Byte offset within a globally allocated backing buffer.
    uint32_t address_offset = 0;
    // Data format of the circular buffer.
    tt::DataFormat data_format = tt::DataFormat::Invalid;
    // Optional: If this CB is overlapped by another CB, this will hold the name of that CB.
    std::optional<Conv2dCb> overlapped_by_cb = std::nullopt;

    uint32_t cb_size_per_core() const { return num_pages * page_size; }
};

// Returns a vector of CBInfo objects for the Conv2d operation.
// The vector will contain information about all circular buffers used in the Conv2d operation.
// When the program factory has the real reader indices DRAM buffer, it can pass its actual page
// size so the predicted READER_INDICES CB footprint matches the CB the factory creates. Auto-shard
// L1 estimation passes std::nullopt and falls back to the worst case (1 uint16 index per output row).
std::vector<CBInfo> get_cb_info(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const Conv2dBlockConfig& block_config,
    const Conv2dParallelizationConfig& pconfig,
    const ttnn::Shape& weights_shape,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> input_shape,
    std::array<uint32_t, 2> dilation,
    const Conv2dConfig& conv_config,
    DataType input_datatype,
    DataType output_datatype,
    std::array<uint32_t, 2> conv_input_shard_shape,
    uint32_t output_image_width,
    bool enable_bias,
    bool is_1d_depthwise_conv,
    bool skip_act_cb_create,
    uint32_t input_channels_padded,
    std::optional<uint32_t> reader_indices_actual_page_size = std::nullopt);

const CBInfo& get_cb_info_by_name(const std::vector<CBInfo>& cb_info, Conv2dCb cb_name);
CBInfo& access_cb_info_by_name(const std::vector<CBInfo>& cb_info, Conv2dCb cb_name);

bool is_split_reader_supported(
    TensorMemoryLayout memory_layout, bool is_1d_depthwise_conv, uint32_t act_block_h_ntiles);

bool is_split_reader_viable(
    TensorMemoryLayout memory_layout,
    uint32_t act_block_h_ntiles,
    uint32_t input_channels_padded,
    uint32_t kernel_width,
    tt::ARCH arch,
    DataType input_datatype,
    uint32_t weights_block_ntiles,
    uint32_t weights_tile_size,
    uint32_t dilation_w,
    uint32_t num_blocks_act_h,
    uint32_t act_block_w_ntiles,
    bool fp32_dest_acc,
    DataType output_datatype,
    bool act_reuse_enabled);

conv_op_l1_usage predicted_conv2d_l1_usage(
    const Conv2dParams& operation_attributes,
    const Conv2dInputs& tensor_args,
    std::optional<uint32_t> reader_indices_actual_page_size);

void post_conv2d_op_memory_checks(
    const tt::tt_metal::distributed::MeshWorkload& workload,
    const Conv2dParams& operation_attributes,
    const Conv2dInputs& tensor_args);

void validate_conv2d_realized_dfb_size(uint32_t predicted_size, uint32_t realized_size);
void validate_conv2d_allocator_delta(
    uint32_t pre_op_size, uint32_t post_op_size, uint32_t predicted_tensor_allocation_size);

}  // namespace ttnn::prim
