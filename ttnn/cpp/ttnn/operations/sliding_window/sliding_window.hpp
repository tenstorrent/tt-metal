// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <string>
#include <tuple>
#include <fmt/core.h>

#include "ttnn/tensor/host_buffer/functions.hpp"

namespace ttnn::operations::sliding_window {

struct ParallelConfig {
    CoreRangeSet grid = {};
    tt::tt_metal::TensorMemoryLayout shard_scheme;
    tt::tt_metal::ShardOrientation shard_orientation;

    bool operator==(const ParallelConfig& other) {
        return (
            grid == other.grid && shard_scheme == other.shard_scheme && shard_orientation == other.shard_orientation);
    }
    bool operator!=(const ParallelConfig& other) { return !(*this == other); }

    std::size_t get_hash() const {
        return std::hash<std::string>{}(
            grid.str() + "_" + std::to_string(int(shard_scheme)) + "_" + std::to_string(int(shard_orientation)));
    }
};

using uint32_pair_t = std::pair<uint32_t, uint32_t>;

std::array<uint32_t, 4> get_pair_n4_padding(
    const std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>& padding);
struct SlidingWindowConfig {
    // input tensor shape
    uint32_t batch_size = 0;
    uint32_t channels = 0;
    uint32_pair_t input_hw = {0, 0};

    // windowing parameters
    uint32_pair_t window_hw = {1, 1};
    uint32_pair_t stride_hw = {1, 1};
    std::array<uint32_t, 4> padding = {0, 0, 0, 0};
    uint32_pair_t output_pad_hw = {0, 0};
    uint32_pair_t dilation_hw = {1, 1};

    // parallel configuration
    uint32_t num_cores_nhw = 1;                                             // num cores along collapsed height nhw
    uint32_t num_cores_c = 1;                                               // num cores along width c
    CoreRangeSet core_range_set = CoreRangeSet(CoreRange({0, 0}, {0, 0}));  // active cores

    bool snap_to_tile = false;
    bool is_bilinear = false;
    bool is_transpose = false;
    bool ceil_mode = false;
    bool is_avg_pool = false;

    std::string to_string() const;
    bool has_parallel_config() const;
    /**
     * Unique hash val for the sliding window configuration.
     */
    std::size_t get_hash() const;

    /**
     * Return the input shape (excluding depth)
     */
    ttnn::Shape get_input_shape() const;

    /**
     * Calculate the window op output shape, excludes the channel dimension since this config is independent of the
     * depth.
     */
    ttnn::Shape get_output_shape() const;

    uint32_t get_pad_h() const;
    uint32_t get_pad_w() const;
    uint32_t get_ceil_pad_h() const;
    uint32_t get_ceil_pad_w() const;

    ttnn::Shape get_transposed_full_input_shape() const;

    std::array<uint32_pair_t, 2> get_transposed_real_padding() const;
    /**
     * Calculate output tensor shard height
     */
    uint32_t get_output_shard_y(bool snap_to_tile = false) const;
};  // struct SlidingWindowConfig

struct Range {
    uint32_t start{0};
    uint32_t end{0};
};

struct ShardBoundary {
    Range output_range;
    Range input_range;
};

struct PixelMetadata {
    bool is_pad{false};
    uint32_t src_core_id{0};
    uint32_t src_local_idx{0};
};

std::vector<bool> generate_pad_metadata(const SlidingWindowConfig& config);

std::vector<uint32_t> generate_op_trace_metadata(const SlidingWindowConfig& config);

std::vector<ShardBoundary> generate_shard_boundaries(
    const SlidingWindowConfig& config, const std::vector<uint32_t>& op_trace_metadata);

std::vector<PixelMetadata> generate_tensor_metadata(
    const std::vector<bool>& pad_metadata, const SlidingWindowConfig& config, uint32_t shard_height);

uint32_t generate_max_out_nsticks_per_core(const std::vector<ShardBoundary>& shard_boundaries);

std::tuple<std::vector<std::vector<std::vector<uint16_t>>>, int> generate_inplace_halo_kernel_config_tensors(
    const std::vector<PixelMetadata>& tensor_metadata,
    const std::vector<ShardBoundary>& shard_boundaries,
    bool is_block_sharded,
    bool transpose_mcast,
    bool remote_read,
    bool is_in_tiled,
    tt::tt_metal::IDevice* device,
    uint32_t max_out_nsticks_per_core = INT_MAX,
    uint32_t in_nsticks_per_core = 0,
    bool in_place = false);

struct HaloGatherKernelConfig {
    std::vector<std::vector<uint16_t>> pad_config0;
    std::vector<std::vector<uint16_t>> pad_config1;
    std::vector<std::vector<uint16_t>> gather_config0;
    std::vector<std::vector<uint16_t>> gather_config1;
    std::vector<uint16_t> number_of_blocks_per_core;
};

HaloGatherKernelConfig generate_halo_kernel_config_tensors(
    const std::vector<PixelMetadata>& tensor_metadata,
    const std::vector<ShardBoundary>& shard_boundaries,
    bool is_block_sharded,
    bool transpose_mcast,
    bool remote_read,
    tt::tt_metal::IDevice* device,
    bool is_in_tiled,
    int block_size);

std::vector<std::vector<uint16_t>> generate_sliding_window_op_config(
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<ShardBoundary>& shard_boundaries,
    uint32_t stride_w,
    bool is_conv =
        false,  // In convs, we have the concept of dividing the act block (act_block_h_override and split reader)
    uint32_t reader0_datums = 0,
    uint32_t reader1_datums = 0,
    bool pad_cores = true);

std::vector<uint16_t> flatten(const std::vector<std::vector<uint16_t>>& input, uint32_t extend_with_zeroes = 0);

uint32_t get_repeat_factor_for_replicating_nhw_config_across_grid(const ParallelConfig& p_config);

std::vector<uint16_t> replicate_config(const std::vector<uint16_t>& config_vector, int factor);

std::vector<uint16_t> remap_nhw_scalar_argument_across_full_grid(
    const std::vector<uint16_t>& config, const ParallelConfig& parallel_config);

Tensor construct_on_host_config_tensor(
    const std::vector<std::vector<uint16_t>>& config, const ParallelConfig& p_config);

Tensor move_config_tensor_to_device(
    const Tensor& config_tensor, const ParallelConfig& p_config, bool is_block_sharded, tt::tt_metal::IDevice* device);

}  // namespace ttnn::operations::sliding_window

// hash and formatter template specializations for config structs

template <>
struct std::hash<ttnn::operations::sliding_window::SlidingWindowConfig> {
    size_t operator()(const ttnn::operations::sliding_window::SlidingWindowConfig& config) const {
        return std::hash<int>()(config.get_hash());
    }
};

template <>
struct std::hash<ttnn::operations::sliding_window::ParallelConfig> {
    size_t operator()(const ttnn::operations::sliding_window::ParallelConfig& config) const {
        return std::hash<int>()(config.get_hash());
    }
};

template <>
struct fmt::formatter<ttnn::operations::sliding_window::SlidingWindowConfig> : formatter<string_view> {
    auto format(const ttnn::operations::sliding_window::SlidingWindowConfig& t, fmt::format_context& ctx) const
        -> format_context::iterator;
};

template <>
struct fmt::formatter<ttnn::operations::sliding_window::ParallelConfig> : formatter<string_view> {
    auto format(const ttnn::operations::sliding_window::ParallelConfig& t, fmt::format_context& ctx) const
        -> format_context::iterator;
};

template <>
struct fmt::formatter<ttnn::operations::sliding_window::ShardBoundary> : formatter<string_view> {
    auto format(const ttnn::operations::sliding_window::ShardBoundary& t, fmt::format_context& ctx) const
        -> format_context::iterator;
};
