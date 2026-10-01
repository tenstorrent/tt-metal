// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>

// Exports symbols
#include <tt-metalium/tensor/tensor_apis.hpp>

namespace ttnn {

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
ttsl::optional_reference<const tt::tt_metal::MeshTensor> as_optional_mesh_tensor(const std::optional<Tensor>& opt);

// Returns the optimal worker cores for a sharded tensor.
std::vector<tt::tt_metal::CoreCoord> get_optimal_worker_cores_for_sharded_tensor(
    const Tensor& tensor, tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default);

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
tt::tt_metal::CBDescriptor cb_descriptor_from_sharded_tensor(
    uint8_t cb_index,
    const Tensor& tensor,
    uint32_t address_offset = 0,
    uint32_t total_size = 0,
    const std::optional<tt::tt_metal::CoreRangeSet>& core_ranges = std::nullopt);

/**
 * @brief Get the L1 byte address a CB descriptor is programmed at.
 *
 * Returns the backing buffer's address + address_offset, or just address_offset when no buffer is set
 * (manually placed CB). A per-core-allocated buffer sits at a different address on each core, so its
 * address is the one on the CB's cores, not Buffer::address() (the first core's); those cores must share
 * one address, and for a tensor-backed descriptor so must every local device, or this TT_FATALs.
 */
inline uint32_t get_cb_address(const tt::tt_metal::CBDescriptor& desc) {
    namespace per_core_allocation = tt::tt_metal::experimental::per_core_allocation;
    auto addr_offset = desc.address_offset;
    const tt::tt_metal::Buffer* buffer = desc.buffer;
    if (buffer == nullptr && desc.tensor != nullptr) {
        buffer = desc.tensor->mesh_buffer().get_reference_buffer();
    }
    if (buffer == nullptr) {
        return addr_offset;
    }
    if (!per_core_allocation::is_per_core_allocation(*buffer) || desc.core_ranges.empty()) {
        return buffer->address() + addr_offset;
    }
    std::optional<tt::tt_metal::DeviceAddr> base;
    for (const auto& core_range : desc.core_ranges.ranges()) {
        for (const auto& core : core_range) {
            const auto address = desc.buffer == nullptr ? per_core_allocation::get_uniform_per_core_address(
                                                              desc.tensor->mesh_buffer(), core)
                                                        : per_core_allocation::get_per_core_address(*buffer, core);
            if (!base.has_value()) {
                base = address;
                continue;
            }
            TT_FATAL(
                address == *base,
                "CB descriptor on cores {} is backed by a per-core-allocated buffer that sits at {:#x} and {:#x} on "
                "different cores; a circular buffer has one address",
                desc.core_ranges.str(),
                *base,
                address);
        }
    }
    return *base + addr_offset;
}

}  // namespace ttnn
