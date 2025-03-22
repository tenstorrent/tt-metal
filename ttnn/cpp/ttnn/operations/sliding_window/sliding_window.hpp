// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
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

struct SlidingWindowConfig {
    // input tensor shape
    uint32_t batch_size = 0;
    uint32_pair_t input_hw = {0, 0};

    // windowing parameters
    uint32_pair_t window_hw = {1, 1};
    uint32_pair_t stride_hw = {1, 1};
    uint32_pair_t pad_hw = {0, 0};
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
    const std::vector<bool>& pad_metadata,
    const SlidingWindowConfig& config,
    uint32_t reshard_num_cores_nhw = 0,
    bool is_in_tiled = true);

uint32_t generate_max_out_nsticks_per_core(const std::vector<ShardBoundary>& shard_boundaries);

struct HaloGatherKernelConfig {
    std::vector<std::vector<uint16_t>> pad_config;
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
    bool is_in_tiled);

std::vector<std::vector<uint16_t>> generate_sliding_window_op_config(
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<ShardBoundary>& shard_boundaries,
    bool pad_tile = false,
    bool pad_cores = false);

std::vector<uint16_t> flatten(const std::vector<std::vector<uint16_t>>& input);

template <typename T>
std::pair<std::vector<T>, uint32_t> replicate_config_across_grid(
    const std::vector<T>& config_vector, const SlidingWindowConfig& sw_config, const ParallelConfig& p_config) {
    if (p_config.shard_scheme == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        return {config_vector, 1};
    } else if (p_config.shard_scheme == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t repeat_factor = p_config.grid.num_cores();
        std::vector<T> repeat_config;
        for (uint32_t i = 0; i < repeat_factor; ++i) {
            repeat_config.insert(repeat_config.end(), config_vector.begin(), config_vector.end());
        }
        return {repeat_config, repeat_factor};
    } else if (p_config.shard_scheme == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
        TT_ASSERT(p_config.grid.ranges().size() == 1, "BLOCK_SHARDED should have just a single core range");
        // NOTE: it is assumed that the range start is always (0, 0)
        uint32_t ncores_y = p_config.grid.ranges().begin()->end_coord.y + 1;
        uint32_t ncores_x = p_config.grid.ranges().begin()->end_coord.x + 1;
        std::vector<T> repeat_config;
        uint32_t repeat_factor = 0;
        if (p_config.shard_orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR) {
            TT_ASSERT(
                config.size() == ncores_y,
                "Invalid config size {} (!= {}) for BLOCK_SHARDED ROW_MAJOR",
                config.size(),
                ncores_y);
            repeat_factor = ncores_x;
        } else if (p_config.shard_orientation == tt::tt_metal::ShardOrientation::COL_MAJOR) {
            TT_ASSERT(
                config.size() == ncores_x,
                "Invalid config size {} (!= {}) for BLOCK_SHARDED COL_MAJOR",
                config.size(),
                ncores_x);
            repeat_factor = ncores_y;
        } else {
            TT_ASSERT(false, "Unsupported shard orientation");
        }
        for (uint32_t i = 0; i < repeat_factor; ++i) {
            repeat_config.insert(repeat_config.end(), config_vector.begin(), config_vector.end());
        }
        return {repeat_config, repeat_factor};
    }
    TT_FATAL(false, "Unsupported shard scheme");
}

Tensor construct_on_host_config_tensor(
    const std::vector<std::vector<uint16_t>>& config,
    const SlidingWindowConfig& sw_config,
    const ParallelConfig& p_config);

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
