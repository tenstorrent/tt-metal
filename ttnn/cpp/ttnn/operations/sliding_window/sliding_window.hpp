// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>
#include <fmt/core.h>

#include "ttnn/tensor/host_buffer/functions.hpp"

namespace ttnn::operations::sliding_window {

struct ParallelConfig {
    CoreRangeSet grid = {{}};
    TensorMemoryLayout shard_scheme;
    ShardOrientation shard_orientation;

    bool operator==(const ParallelConfig &other) {
        return (grid == other.grid && shard_scheme == other.shard_scheme && shard_orientation == other.shard_orientation);
    }
    bool operator!=(const ParallelConfig &other) {
        return !(*this == other);
    }

    std::size_t get_hash() const {
        return std::hash<std::string>{}(grid.str() + "_" + std::to_string(int(shard_scheme)) + "_" + std::to_string(int(shard_orientation)));
    }
};

using uint32_pair_t = std::pair<uint32_t, uint32_t>;

struct SlidingWindowConfig {

    // input tensor shape
    uint32_t batch_size = 0;
    uint32_pair_t input_hw = {0, 0};

    // windowing parameters
    uint32_pair_t window_hw  = {1, 1};
    uint32_pair_t stride_hw  = {1, 1};
    uint32_pair_t pad_hw = {0, 0} ;
    uint32_pair_t dilation_hw = {1, 1};

    // parallel configuration
    uint32_t num_cores_nhw = 1;        // num cores along collapsed height nhw
    CoreRangeSet core_range_set = std::set{CoreRange({0, 0}, {0, 0})};   // active cores

    bool snap_to_tile = false;

    std::string to_string() const;
    bool has_parallel_config() const;
    /**
        * Unique hash val for the sliding window configuration.
        */
    std::size_t get_hash() const;

    /**
        * Return the input shape (excluding depth)
        */
    Shape get_input_shape() const;

    /**
        * Calculate the window op output shape, excludes the channel dimension since this config is independent of the depth.
        */
    Shape get_output_shape() const;

    /**
        * Calculate output tensor shard height
        */
    uint32_t get_output_shard_y(bool snap_to_tile = false) const;
}; // struct SlidingWindowConfig


std::vector<bool> generate_pad_metadata(const SlidingWindowConfig& config);
std::vector<uint32_t> generate_op_trace_metadata(const SlidingWindowConfig& config);
std::vector<std::pair<uint32_pair_t, uint32_pair_t>> generate_shard_boundaries(const SlidingWindowConfig& config, const std::vector<uint32_t>& op_trace_metadata);
std::vector<std::pair<bool, uint32_pair_t>> generate_tensor_metadata(const std::vector<bool>& pad_metadata, const SlidingWindowConfig& config, uint32_t reshard_num_cores_nhw = 0, bool is_in_tiled = true);
uint32_t generate_max_out_nsticks_per_core(const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries);
std::tuple<std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>> generate_halo_kernel_config_tensors(const std::vector<std::pair<bool, uint32_pair_t>>& tensor_metadata, const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries, bool is_block_sharded, bool transpose_mcast, bool remote_read, Device* device);
std::vector<std::vector<uint16_t>> generate_sliding_window_op_config(const std::vector<uint32_t>& op_trace_metadata, const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries, bool pad_tile = false, bool pad_last_core = false);
std::vector<uint16_t> flatten(const std::vector<std::vector<uint16_t>>& input);
Tensor construct_on_host_config_tensor(const std::vector<std::vector<uint16_t>>& config, const SlidingWindowConfig& sw_config, const ParallelConfig& p_config);
Tensor move_config_tensor_to_device(const Tensor& config_tensor, const ParallelConfig& p_config, bool is_block_sharded, Device* device);

} // namespace ttnn::operations::sliding_window

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

template <> struct fmt::formatter<ttnn::operations::sliding_window::SlidingWindowConfig >: formatter<string_view> {
    auto format(const ttnn::operations::sliding_window::SlidingWindowConfig& t, fmt::format_context& ctx) const
    -> format_context::iterator;
};

template <> struct fmt::formatter<ttnn::operations::sliding_window::ParallelConfig>: formatter<string_view> {
    auto format(const ttnn::operations::sliding_window::ParallelConfig& t, fmt::format_context& ctx) const
    -> format_context::iterator;
};
