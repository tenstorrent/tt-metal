// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <fmt/core.h>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "tt-metalium/hal.hpp"

namespace ttnn::operations::sliding_window {

// Values are part of the serialized program-cache key via SlidingWindowConfig::to_string().
// Keep existing values stable; only append new modes.
enum class PaddingMode : uint8_t {
    Zeros = 0,
    Replicate = 1,
};

struct ParallelConfig {
    CoreRangeSet grid;
    tt::tt_metal::TensorMemoryLayout shard_scheme{0};
    tt::tt_metal::ShardOrientation shard_orientation{0};

    bool operator==(const ParallelConfig& other) const {
        return (
            grid == other.grid && shard_scheme == other.shard_scheme && shard_orientation == other.shard_orientation);
    }
    bool operator!=(const ParallelConfig& other) const { return !(*this == other); }

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
    std::optional<uint32_pair_t> ceil_pad_hw = std::nullopt;

    // bilinear scaling parameters
    uint32_t scale_h = 1;
    uint32_t scale_w = 1;

    // parallel configuration
    uint32_t num_cores_nhw = 1;                                             // num cores along collapsed height nhw
    uint32_t num_cores_c = 1;                                               // num cores along width c
    CoreRangeSet core_range_set = CoreRangeSet(CoreRange({0, 0}, {0, 0}));  // active cores

    bool snap_to_tile = false;
    bool is_bilinear = false;
    bool is_transpose = false;
    bool ceil_mode = false;
    PaddingMode padding_mode = PaddingMode::Zeros;

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

    uint32_t get_pad_top() const;
    uint32_t get_pad_bottom() const;
    uint32_t get_pad_left() const;
    uint32_t get_pad_right() const;
    uint32_t get_pad_h() const;
    uint32_t get_pad_w() const;
    uint32_t get_ceil_pad_h() const;
    uint32_t get_ceil_pad_w() const;
    uint32_pair_t get_ceil_pad_hw() const;

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

std::vector<uint32_t> generate_op_trace_metadata_bilinear(const SlidingWindowConfig& config);

std::pair<uint32_t, uint32_t> find_minmax_trace_indices(
    const std::vector<uint32_t>& op_trace_metadata, uint32_t start_idx, uint32_t end_idx);

std::vector<ShardBoundary> generate_shard_boundaries(const SlidingWindowConfig& config);

std::vector<PixelMetadata> generate_tensor_metadata(
    const std::vector<bool>& pad_metadata, const SlidingWindowConfig& config, uint32_t shard_height);

uint32_t generate_max_out_nsticks_per_core(const std::vector<ShardBoundary>& shard_boundaries);

uint32_t calculate_precise_halo_output_elems(
    const SlidingWindowConfig& config, const std::array<uint32_t, 2>& shard_shape);

// Max outbound-halo sticks any single core sends to other cores. This sizes the in-place
// halo remote-temp CB (== max_ref_size) and drives the in-place net-L1 decision.
uint32_t compute_max_outbound_halo_sticks(
    const std::vector<PixelMetadata>& tensor_metadata,
    const std::vector<ShardBoundary>& shard_boundaries,
    uint32_t num_cores_nhw);

// Silent in-place-halo activation decision. Returns true iff running halo in-place (output
// shard buffer overlapping the input shard buffer) would net-save L1 AND the layout is a
// class validated to be corruption-safe. Pure function of the attributes so the halo op,
// its program factory, and the pool/conv callers all reach the same decision without
// storing state. See IN_PLACE_HALO_REDO.md section 10. NOTE: there is intentionally no
// user-facing toggle -- in-place activates automatically only when it is a clear win.
// `allow_in_place` is a per-caller capability gate: it must be true for in-place to be
// considered at all (only the pool caller opts in; conv/upsample/fold pass false).
// `memory_layout` selects the validated corruption-safe classes: HEIGHT_SHARDED,
// WIDTH_SHARDED and BLOCK_SHARDED are all supported (width-sharded halo is all-local so
// max_ref_size==0; block-sharded uses the transpose_mcast / column-major NOC orientation).
// `input_shard_width_bytes` = per-core shard width in bytes (shard_spec.shape[1] * datum_size); used by
// the Blackhole alignment-safety gate for row-major input (see .cpp). Callers pass the input tensor's
// shard width; pass 0 to decline in-place when it is unknown.
bool should_halo_be_in_place(
    bool allow_in_place,
    const SlidingWindowConfig& config,
    uint32_t in_nsticks_per_core,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    bool is_in_tiled,
    uint32_t input_shard_width_bytes);

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
    uint32_t num_cores_x,
    bool is_in_tiled,
    int block_size);

// In-place halo config generator. Self-contained sibling of generate_halo_kernel_config_tensors:
// it emits the SEPARATE flat-uint16 layout consumed by the in-place halo device kernel and returns
// the max remote-temp reference size (max_ref_size) used to size the remote-temp CB. The LOCAL
// config uses a forward-then-reverse entry ordering that pairs with the kernel's overlapping-copy
// direction logic; do not reorder. `device` is a MeshDevice* so callers can reuse the same handle
// they pass to move_config_tensor_to_device(). Returns {config, max_ref_size} where config is the
// 6-vector {pad0, pad1, local0, local1, remote0, remote1}.
std::tuple<std::vector<std::vector<std::vector<uint16_t>>>, int> generate_inplace_halo_kernel_config_tensors(
    const std::vector<PixelMetadata>& tensor_metadata,
    const std::vector<ShardBoundary>& shard_boundaries,
    bool is_block_sharded,
    bool transpose_mcast,
    bool remote_read,
    bool is_in_tiled,
    tt::tt_metal::distributed::MeshDevice* device,
    uint32_t num_cores_x,
    uint32_t max_out_nsticks_per_core,
    uint32_t in_nsticks_per_core,
    bool in_place,
    uint32_t in_out_shard_size_delta);

void visualize_sliding_window_op_config(const std::vector<std::vector<uint16_t>>& config);

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
    const std::vector<std::vector<uint16_t>>& config, const ParallelConfig& p_config, bool store_in_dram = false);

Tensor move_config_tensor_to_device(
    const Tensor& config_tensor,
    const ParallelConfig& p_config,
    bool is_block_sharded,
    tt::tt_metal::distributed::MeshDevice* device,
    bool store_in_dram = false);

uint32_t align_buffer(uint32_t size);

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
