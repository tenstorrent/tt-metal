// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>
#include <fmt/core.h>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "utils.hpp"

namespace tt::tt_metal {

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
        uint32_t batch_size_;
        uint32_pair_t input_hw_;

        // windowing parameters
        uint32_pair_t window_hw_;
        uint32_pair_t stride_hw_;
        uint32_pair_t pad_hw_;
        uint32_pair_t dilation_hw_;

        // parallel configuration
        bool has_parallel_config_;
        uint32_t num_cores_nhw_;        // num cores along collapsed height nhw
        CoreRangeSet core_range_set_;   // active cores

        bool snap_to_tile_;

        SlidingWindowConfig(uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t window_h, uint32_t window_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t dilation_h = 1, uint32_t dilation_w = 1, uint32_t num_cores_nhw = 0, CoreRangeSet core_range = {{}}, bool snap_to_tile = false)
            : batch_size_(batch_size), input_hw_(input_h, input_w), window_hw_(window_h, window_w), stride_hw_(stride_h, stride_w), pad_hw_(pad_h, pad_w), dilation_hw_(dilation_h, dilation_w), has_parallel_config_(false), num_cores_nhw_(num_cores_nhw), core_range_set_(core_range), snap_to_tile_(snap_to_tile) {
                has_parallel_config_ = num_cores_nhw_ > 0 && !core_range_set_.ranges().empty();
            }

        /**
         * Unique hash val for the sliding window configuration.
         */
        std::size_t get_hash() const {
            return std::hash<std::string>{}(std::to_string(batch_size_)
                                            + "_" + std::to_string(std::get<0>(input_hw_)) + "_" + std::to_string(std::get<1>(input_hw_))
                                            + "_" + std::to_string(std::get<0>(window_hw_)) + "_" + std::to_string(std::get<1>(window_hw_))
                                            + "_" + std::to_string(std::get<0>(stride_hw_)) + "_" + std::to_string(std::get<1>(stride_hw_))
                                            + "_" + std::to_string(std::get<0>(pad_hw_)) + "_" + std::to_string(std::get<1>(pad_hw_))
                                            + "_" + std::to_string(std::get<0>(dilation_hw_)) + "_" + std::to_string(std::get<1>(dilation_hw_))
                                            + "_" + std::to_string(num_cores_nhw_) + "_" + core_range_set_.str());
        }

        /**
         * Return the input shape (excluding depth)
         */
        Shape get_input_shape() const {
            return Shape({batch_size_, std::get<0>(input_hw_), std::get<1>(input_hw_)});
        }

        /**
         * Calculate the window op output shape, excludes the channel dimension since this config is independent of the depth.
         */
        Shape get_output_shape() const {
            uint32_t output_h = (input_hw_.first + 2 * pad_hw_.first - dilation_hw_.first * window_hw_.first) / stride_hw_.first + 1;
            uint32_t output_w = (input_hw_.second + 2 * pad_hw_.second - dilation_hw_.second * window_hw_.second) / stride_hw_.second + 1;
            // uint32_t output_h = (std::get<0>(input_hw_) + 2 * std::get<0>(pad_hw_) - std::get<0>(dilation_hw_) * std::get<0>(window_hw_)) / std::get<0>(stride_hw_) + 1;
            // uint32_t output_w = (std::get<1>(input_hw_) + 2 * std::get<1>(pad_hw_) - std::get<1>(dilation_hw_) * std::get<1>(window_hw_)) / std::get<1>(stride_hw_) + 1;
            log_debug(LogOp, "output_size: {} {} {}", batch_size_, output_h, output_w);
            return Shape({batch_size_, output_h, output_w, 0});
        }

        /**
         * Calculate output tensor shard height
         */
        uint32_t get_output_shard_y(bool snap_to_tile = false) const {
            TT_ASSERT(has_parallel_config_, "Parallel config is not set in SlidingWindowConfig");
            Shape output_shape = get_output_shape();
            uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
            uint32_t output_nhw_padded = round_up(output_nhw, num_cores_nhw_ * (snap_to_tile ? constants::TILE_HEIGHT : 1));
            log_debug(LogOp, "output_nhw: {} output_nhw_padded: {} num_cores_nhw: {}", output_nhw, output_nhw_padded, num_cores_nhw_);
            return (output_nhw_padded / num_cores_nhw_);
        }
    }; // struct SlidingWindowConfig

    namespace sliding_window {

        std::vector<bool> generate_pad_metadata(const SlidingWindowConfig& config);
        std::vector<uint32_t> generate_op_trace_metadata(const SlidingWindowConfig& config);
        std::vector<std::pair<uint32_pair_t, uint32_pair_t>> generate_shard_boundaries(const SlidingWindowConfig& config, const std::vector<uint32_t>& op_trace_metadata);
        std::vector<std::pair<bool, uint32_pair_t>> generate_tensor_metadata(const std::vector<bool>& pad_metadata, const SlidingWindowConfig& config, uint32_t reshard_num_cores_nhw = 0);
        uint32_t generate_max_out_nsticks_per_core(const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries);
        std::tuple<std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>> generate_halo_kernel_config_tensors(const std::vector<std::pair<bool, uint32_pair_t>>& tensor_metadata, const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries, bool is_block_sharded, bool transpose_mcast, bool remote_read, Device* device);
        std::vector<std::vector<uint16_t>> generate_sliding_window_op_config(const std::vector<uint32_t>& op_trace_metadata, const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries, bool pad_tile = false, bool pad_last_core = false);
        std::vector<uint16_t> flatten(const std::vector<std::vector<uint16_t>>& input);
        Tensor construct_on_host_config_tensor(const std::vector<std::vector<uint16_t>>& config, const SlidingWindowConfig& sw_config, const ParallelConfig& p_config);
        Tensor move_config_tensor_to_device(const Tensor& config_tensor, const ParallelConfig& p_config, bool is_block_sharded, Device* device);
    } // namespace sliding_window
} // namespace tt::tt_metal

// hash and formatter template specializations for config structs

template <>
struct std::hash<tt::tt_metal::SlidingWindowConfig> {
    size_t operator()(const tt::tt_metal::SlidingWindowConfig& config) const {
        return std::hash<int>()(config.get_hash());
    }
};

template <>
struct std::hash<tt::tt_metal::ParallelConfig> {
    size_t operator()(const tt::tt_metal::ParallelConfig& config) const {
        return std::hash<int>()(config.get_hash());
    }
};

template <> struct fmt::formatter<tt::tt_metal::SlidingWindowConfig>: formatter<string_view> {
    auto format(const tt::tt_metal::SlidingWindowConfig& t, fmt::format_context& ctx) {
        // std::string str = fmt::format("SlidingWindowConfig(batch_size_={}, input_hw_={}, window_hw_={}, stride_hw_={}, pad_hw_={}, dilation_hw_={}, num_cores_nhw_={}, core_range_set_=)",
        std::string str = fmt::format("SlidingWindowConfig(batch_size_={}, input_hw_=({},{}), window_hw_=({},{}), stride_hw_=({},{}), pad_hw_=({},{}), dilation_hw_=({},{}), num_cores_nhw_={}, core_range_set_={})",
            t.batch_size_,
            t.input_hw_.first,
            t.input_hw_.second,
            t.window_hw_.first,
            t.window_hw_.second,
            t.stride_hw_.first,
            t.stride_hw_.second,
            t.pad_hw_.first,
            t.pad_hw_.second,
            t.dilation_hw_.first,
            t.dilation_hw_.second,
            t.num_cores_nhw_,
            t.core_range_set_.str());
        return fmt::format_to(ctx.out(), "{}", str);
    }
};

template <> struct fmt::formatter<tt::tt_metal::ParallelConfig>: formatter<string_view> {
    auto format(const tt::tt_metal::ParallelConfig& t, fmt::format_context& ctx) {
        std::string str = fmt::format("ParallelConfig(grid={}, shard_scheme={}, shard_orientation={})", t.grid.str(), int(t.shard_scheme), int(t.shard_orientation));
        return fmt::format_to(ctx.out(), "{}", str);
    }
};
