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

// Returns an optional_reference to the underlying MeshTensor of `opt`.
//
// - If `opt` is empty, returns an empty optional_reference.
// - If `opt` holds a device tensor, returns a reference to its MeshTensor.
// - If `opt` holds a non-device (host) tensor, TT_FATALs.
//
// The returned reference borrows from the Tensor inside `opt`; the caller must
// keep `opt` alive for as long as the returned reference is used.
ttsl::optional_reference<const MeshTensor> as_optional_mesh_tensor(const std::optional<Tensor>& opt);

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
    auto addr_offset = desc.address_offset;
    if (desc.buffer != nullptr) {
        return desc.buffer->address() + addr_offset;
    }
    if (desc.tensor != nullptr) {
        return desc.tensor->address() + addr_offset;
    }
    return addr_offset;
}

}  // namespace tt::tt_metal
