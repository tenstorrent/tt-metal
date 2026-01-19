// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace tt::tt_metal {

// Returns true if the logical tensor data matches the physical tensor data:
// 1. Row major layout is used.
// 2. Logical 2D shape matches physical shape.
// Used for optimizing conversion operations.
bool logical_matches_physical(const TensorSpec& tensor_spec);

// Returns true if tensor has Host storage.
bool is_cpu_tensor(const Tensor& tensor);

// Returns true if tensor is on device.
bool is_device_tensor(const Tensor& tensor);

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
CBDescriptor cb_descriptor_from_sharded_tensor(uint8_t cb_index, const Tensor& tensor);

}  // namespace tt::tt_metal
