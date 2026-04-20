// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

// Exports symbols
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

namespace tt::tt_metal {

// Returns true if tensor has Host storage.
bool is_cpu_tensor(const Tensor& tensor);

// Returns true if tensor is on device.
bool is_device_tensor(const Tensor& tensor);

// Returns the optimal worker cores for a sharded tensor.
std::vector<CoreCoord> get_optimal_worker_cores_for_sharded_tensor(
    const Tensor& tensor, NOC noc = NOC::RISCV_0_default);

/**
 * @brief Creates a CBDescriptor from a sharded tensor.
 *
 * This function simplifies CB creation for sharded tensors by automatically deriving:
 * - total_size: From tensor's packed buffer size
 * - core_ranges: From tensor's shard spec grid
 * - format_descriptors: From CB index, tensor dtype, and page size
 * - buffer: From tensor's buffer pointer
 *
 * @param cb_index The CB ID to use for this circular buffer
 * @param tensor The sharded tensor to derive CB configuration from
 * @param address_offset Byte offset from buffer base address for CB placement (default 0)
 * @param total_size Total CB size in bytes (default 0 = use tensor's full bank size)
 * @param core_ranges Optional CoreRangeSet override; if std::nullopt, uses the tensor's shard grid
 * @return CBDescriptor with all fields populated from the tensor
 *
 * Example usage (replaces manual calculation of all CB fields):
 * @code
 *   // Old way (manual):
 *   auto act_df = datatype_to_dataformat_converter(device_input_tensor.dtype());
 *   uint32_t tile_size = tt::tile_size(act_df);
 *   uint32_t page_size = round_up_to_mul32(tile_size);
 *   uint32_t num_tiles = calculate_tiles_from_shard(...);
 *   CBDescriptor cb = {
 *       .total_size = num_tiles * page_size,
 *       .core_ranges = all_cores,
 *       .format_descriptors = {{in_cb_id, act_df, page_size}},
 *       .buffer = device_input_tensor.buffer(),
 *   };
 *
 *   // New way (automatic):
 *   CBDescriptor cb = cb_descriptor_from_sharded_tensor(in_cb_id, device_input_tensor);
 * @endcode
 */
CBDescriptor cb_descriptor_from_sharded_tensor(
    uint8_t cb_index,
    const Tensor& tensor,
    uint32_t address_offset = 0,
    uint32_t total_size = 0,
    const std::optional<CoreRangeSet>& core_ranges = std::nullopt);

/**
 * @brief Get the L1 byte address of a CB descriptor.
 *
 * Returns buffer->address() + address_offset when a buffer is present,
 * or just address_offset when no buffer is set (manually placed CB).
 */
inline uint32_t get_cb_address(const CBDescriptor& desc) {
    if (desc.buffer == nullptr) {
        return desc.address_offset;
    }
    return desc.buffer->address() + desc.address_offset;
}

/**
 * @brief Automatically computes a ShardSpec for a sharded MemoryConfig that is missing one.
 *
 * This is a general-purpose utility that can be used by any operation (unary, binary, matmul, etc.)
 * to enable simplified sharding APIs where users pass e.g. L1_HEIGHT_SHARDED_MEMORY_CONFIG without
 * manually computing shard shapes, core grids, and orientations.
 *
 * The function:
 * 1. Returns the config unchanged if it's not sharded or already has a shard_spec.
 * 2. Reuses the input tensor's shard_spec if available.
 * 3. Otherwise, computes an optimal shard_spec based on the tensor shape, device grid, and
 *    memory layout (WIDTH_SHARDED, HEIGHT_SHARDED, or BLOCK_SHARDED).
 *
 * Shard shape computation respects:
 * - Tile alignment (32x32)
 * - L1 alignment constraints
 * - Device core grid limits
 * - Even shard preference (avoids uneven shards for better perf)
 * - L1 memory capacity (fatals if computed shard exceeds per-core L1 size)
 *
 * @param input_tensor  The input tensor (must be on device with non-null device pointer;
 *                       fatals with a descriptive message if either condition is violated)
 * @param output_memory_config  The desired output memory config (possibly missing shard_spec)
 * @return MemoryConfig with shard_spec filled in if needed, otherwise the original config
 *
 * @note BLOCK_SHARDED always uses COL_MAJOR orientation, mapping height shards to grid.x
 *       and width shards to grid.y.
 *
 * Example usage:
 * @code
 *   // In any op's compute_output_specs():
 *   auto resolved_config = compute_auto_shard_spec(input_tensor, args.output_memory_config);
 *   // resolved_config now has a valid shard_spec for sharded configs
 * @endcode
 */
MemoryConfig compute_auto_shard_spec(const Tensor& input_tensor, const MemoryConfig& output_memory_config);

/**
 * @brief Adjusts an existing ShardSpec to match a new tensor shape.
 *
 * Scales shard dimensions proportionally based on the ratio of from_shape to to_shape.
 * Used when inheriting a shard spec from an input tensor whose shape differs from the output.
 */
ShardSpec adjust_shard_spec_to_shape(
    const ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

/**
 * @brief Binary op overload: computes auto shard spec considering two input tensors.
 *
 * Priority: explicit shard_spec > inherit from input_a > inherit from input_b > generate fresh.
 * When inheriting, adjusts the shard spec to match the broadcasted output shape.
 */
MemoryConfig compute_auto_shard_spec(
    const Tensor& input_a,
    const Tensor& input_b,
    const ttnn::Shape& output_shape,
    const MemoryConfig& output_memory_config);

/**
 * @brief Ternary op overload: computes auto shard spec considering three input tensors.
 *
 * Priority: explicit shard_spec > inherit from input with largest shard grid > generate fresh.
 * When inheriting, adjusts the shard spec to match the output shape.
 */
MemoryConfig compute_auto_shard_spec(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    const ttnn::Shape& output_shape,
    const MemoryConfig& output_memory_config);

}  // namespace tt::tt_metal
