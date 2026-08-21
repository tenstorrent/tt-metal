// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>

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
 * @brief Creates a Metal 2.0 DataflowBufferSpec from a sharded tensor.
 *
 * The Metal 2.0 counterpart of cb_descriptor_from_sharded_tensor(). Every entry-format field of
 * DataflowBufferSpec is derived from the tensor instead of being hand-typed next to a
 * TensorParameter that already knows it:
 * - entry_size:            the shard's page stride in L1 (Buffer::aligned_page_size()), i.e. a tile
 *                          for a TILE-layout tensor and a stick for a ROW_MAJOR one. Always derived;
 *                          there is deliberately no override, since a wrong entry_size silently
 *                          corrupts data and is the whole reason this helper exists.
 * - num_entries:           pages per shard (Buffer::aligned_size_per_bank() / aligned_page_size()),
 *                          unless overridden (see @p num_entries).
 * - data_format_metadata:  from the tensor's dtype.
 * - tile_format_metadata:  the tensor's Tile, for any tile-paged DFB.
 * - borrowed_from:         the caller-supplied TensorParameter name (see @p borrowed_from).
 *
 * @param unique_id     The DFB's name within the ProgramSpec.
 * @param tensor        An allocated, sharded tensor to derive the DFB's entry format from.
 * @param num_entries   Number of DFB entries; 0 (default) means "one per page of the shard", i.e.
 *                      the DFB spans the whole resident shard. Pass a value to build a shallower
 *                      DFB (e.g. a double-buffered staging DFB that only borrows the tensor's
 *                      entry format). Mirrors cb_descriptor_from_sharded_tensor's total_size=0.
 * @param borrowed_from Name of the TensorParameter whose L1 memory backs this DFB, or std::nullopt
 *                      (default) for a normally-allocated DFB that merely takes its entry format
 *                      from the tensor. A Tensor does not know the TensorParameter name the
 *                      ProgramSpec author gave it, so this cannot be defaulted for you.
 * @param page_as_tile  Treat the DFB's entries as tiles even when the tensor is ROW_MAJOR: the
 *                      entry becomes one tile (or the whole shard, when the shard is smaller than
 *                      a tile) rather than one stick, and tile_format_metadata is set. This folds
 *                      in the mutation the ProgramDescriptor callers apply by hand right after
 *                      building a CBDescriptor (set_cb_page_size_for_tile). No-op for a TILE tensor,
 *                      which is already tile-paged.
 * @param unpack_face_geometry Passed straight through to
 *                      DataflowBufferSpec::unpack_face_geometry_metadata. Required by the compute
 *                      engine when an entry holds fewer/shorter faces than a full tile (which is
 *                      exactly the sub-tile shard case @p page_as_tile falls back to). NOT derived:
 *                      whether a partial shard is to be unpacked as a partial tile is the kernel
 *                      author's decision, not something the tensor states.
 * @param advanced_options Passed through to DataflowBufferSpec::advanced_options.
 *
 * @return DataflowBufferSpec with all entry-format fields populated from the tensor.
 */
tt::tt_metal::experimental::DataflowBufferSpec dfb_spec_from_sharded_tensor(
    tt::tt_metal::experimental::DFBSpecName unique_id,
    const Tensor& tensor,
    uint32_t num_entries = 0,
    const std::optional<tt::tt_metal::experimental::TensorParamName>& borrowed_from = std::nullopt,
    bool page_as_tile = false,
    const std::optional<tt::tt_metal::FaceGeometry>& unpack_face_geometry = std::nullopt,
    tt::tt_metal::experimental::DFBAdvancedOptions advanced_options = {});

/**
 * @brief Get the L1 byte address of a CB descriptor.
 *
 * Returns buffer->address() + address_offset when a buffer is present,
 * or just address_offset when no buffer is set (manually placed CB).
 */
inline uint32_t get_cb_address(const tt::tt_metal::CBDescriptor& desc) {
    auto addr_offset = desc.address_offset;
    if (desc.buffer != nullptr) {
        return desc.buffer->address() + addr_offset;
    }
    if (desc.tensor != nullptr) {
        return desc.tensor->address() + addr_offset;
    }
    return addr_offset;
}

}  // namespace ttnn
