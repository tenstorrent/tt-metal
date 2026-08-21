// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"

#include <tt_stl/overloaded.hpp>

#include "ttnn/tensor/types.hpp"

#include <tracy/Tracy.hpp>

namespace ttnn {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::datatype_to_dataformat_converter;
using tt::tt_metal::FaceGeometry;
using tt::tt_metal::Layout;
using tt::tt_metal::MeshTensor;
using tt::tt_metal::NOC;
using tt::tt_metal::Tile;
using tt::tt_metal::TileDescriptor;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBAdvancedOptions;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::TensorParamName;

bool is_cpu_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::HOST; }

bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

ttsl::optional_reference<const MeshTensor> as_optional_mesh_tensor(const std::optional<Tensor>& opt) {
    if (opt.has_value()) {
        TT_FATAL(
            is_device_tensor(*opt), "as_optional_mesh_tensor: expected device tensor, got {}", opt->storage_type());
        return {opt->mesh_tensor()};
    }
    return std::nullopt;
}

CBDescriptor cb_descriptor_from_sharded_tensor(
    uint8_t cb_index,
    const Tensor& tensor,
    uint32_t address_offset,
    uint32_t total_size,
    const std::optional<CoreRangeSet>& core_ranges) {
    TT_FATAL(tensor.is_sharded(), "Tensor must be sharded to automatically create a CBDescriptor");
    TT_FATAL(
        (address_offset + total_size) <= tensor.buffer()->aligned_size_per_bank(),
        "Address offset + total size exceeds buffer size");

    uint32_t effective_total_size = (total_size != 0) ? total_size : tensor.buffer()->aligned_size_per_bank();

    return CBDescriptor{
        .total_size = effective_total_size,
        .core_ranges = core_ranges.value_or(tensor.shard_spec()->grid),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = cb_index,
            .data_format = datatype_to_dataformat_converter(tensor.tensor_spec().tensor_layout().get_data_type()),
            .page_size = tensor.buffer()->aligned_page_size(),
            .tile = TileDescriptor(tensor.tensor_spec().tile())}},
        .buffer = tensor.buffer(),
        .address_offset = address_offset,
        .global_circular_buffer = nullptr};
}

DataflowBufferSpec dfb_spec_from_sharded_tensor(
    DFBSpecName unique_id,
    const Tensor& tensor,
    uint32_t num_entries,
    const std::optional<TensorParamName>& borrowed_from,
    bool page_as_tile,
    const std::optional<FaceGeometry>& unpack_face_geometry,
    DFBAdvancedOptions advanced_options) {
    TT_FATAL(
        tensor.is_sharded(),
        "DFB '{}': tensor must be sharded to derive a DataflowBufferSpec from it. An interleaved tensor has no "
        "per-node resident shard for a DFB to describe or borrow.",
        unique_id);
    TT_FATAL(
        tensor.is_allocated(),
        "DFB '{}': tensor must be allocated on device; the shard's page stride and per-bank size come from its "
        "Buffer.",
        unique_id);
    if (borrowed_from.has_value()) {
        // The ProgramSpec validator re-checks this against the TensorSpec, and AttachBorrowedDFBBuffers
        // re-checks it against the Buffer. Checking here too names the offending tensor at build time.
        TT_FATAL(
            tensor.memory_config().is_l1(),
            "DFB '{}' would borrow memory from TensorParameter '{}', but the tensor is not L1-resident. Only L1 "
            "memory can back a DFB.",
            unique_id,
            *borrowed_from);
    }

    const auto& spec = tensor.tensor_spec();
    const auto data_format = datatype_to_dataformat_converter(spec.data_type());
    const Tile tile = spec.tile();

    const tt::tt_metal::Buffer* buffer = tensor.buffer();
    // The shard's L1 footprint, and the stride between consecutive pages inside it. For a TILE
    // tensor a page is a tile; for a ROW_MAJOR one it is a stick. aligned_size_per_bank() is always
    // an exact multiple of aligned_page_size() (see detail::calculate_bank_size_spread).
    const uint32_t shard_bytes = static_cast<uint32_t>(buffer->aligned_size_per_bank());
    const uint32_t page_bytes = static_cast<uint32_t>(buffer->aligned_page_size());
    TT_FATAL(page_bytes > 0, "DFB '{}': tensor's buffer has a zero page size.", unique_id);

    uint32_t entry_size = 0;
    uint32_t derived_entries = 0;
    std::optional<Tile> tile_format = std::nullopt;

    if (spec.layout() == Layout::TILE) {
        // Already tile-paged: one entry per tile. page_as_tile is redundant here, and accepted so a
        // caller can pass it uniformly across a tiled weight and a row-major activation.
        entry_size = page_bytes;
        derived_entries = shard_bytes / page_bytes;
        tile_format = tile;
    } else if (!page_as_tile) {
        // Row-major: the natural entry is a stick, and the DFB is a data-movement conduit. No tile
        // format is claimed - a stick is not a tile. Do not bind such a DFB to a compute kernel;
        // pass page_as_tile to get a tile-paged view of the same shard instead.
        entry_size = page_bytes;
        derived_entries = shard_bytes / page_bytes;
    } else {
        // Row-major memory read as tiles. Mirrors set_cb_page_size_for_tile() in the deepseek gate
        // ProgramDescriptor builders, including its sub-tile fallback: when the shard does not hold a
        // whole number of tiles it becomes a single partial-tile entry, which the compute engine can
        // only unpack correctly with a matching unpack_face_geometry.
        const uint32_t tile_bytes = tile.get_tile_size(data_format);
        TT_FATAL(tile_bytes > 0, "DFB '{}': the tensor's tile has a zero size.", unique_id);
        if (shard_bytes % tile_bytes == 0) {
            entry_size = tile_bytes;
            derived_entries = shard_bytes / tile_bytes;
        } else {
            entry_size = shard_bytes;
            derived_entries = 1;
        }
        tile_format = tile;
    }

    const uint32_t effective_entries = (num_entries != 0) ? num_entries : derived_entries;
    TT_FATAL(
        effective_entries > 0,
        "DFB '{}': derived num_entries is 0. The tensor's shard ({} B) is smaller than one entry ({} B).",
        unique_id,
        shard_bytes,
        entry_size);
    if (borrowed_from.has_value()) {
        const uint64_t dfb_bytes = static_cast<uint64_t>(entry_size) * static_cast<uint64_t>(effective_entries);
        TT_FATAL(
            dfb_bytes <= shard_bytes,
            "DFB '{}' (entry_size {} * num_entries {} = {} B) does not fit in the shard it borrows from "
            "TensorParameter '{}' ({} B per bank).",
            unique_id,
            entry_size,
            effective_entries,
            dfb_bytes,
            *borrowed_from,
            shard_bytes);
    }

    return DataflowBufferSpec{
        .unique_id = std::move(unique_id),
        .entry_size = entry_size,
        .num_entries = effective_entries,
        .data_format_metadata = data_format,
        .tile_format_metadata = tile_format,
        .unpack_face_geometry_metadata = unpack_face_geometry,
        .borrowed_from = borrowed_from,
        .advanced_options = std::move(advanced_options)};
}

std::vector<CoreCoord> get_optimal_worker_cores_for_sharded_tensor(const Tensor& tensor, NOC noc) {
    /**
    This function takes in a sharded device tensor (can be legacy 2D sharded or ND sharded) and returns the optimal
    worker cores to launch programs on for the tensor.

    If the tensor is L1 sharded, the function returns a vector of CoreCoords of all the cores that have shards on them
    in order (based on if the shard orientation is in row or column major order).

    If the tensor is DRAM sharded, the function returns a vector of CoreCoords in order (based on shard orientation) of
    the optimal worker core for each DRAM bank with shards.

    The intended use for this API is inside sharded program factories to get the optimal worker cores to launch the
    program and kernels on. Since the core grid provided in the shard_spec and nd_shard_spec may be larger than the
    number of shards that exist, not all cores in the core grid will have shards on them. This API returns the cores
    that have shards on them in order (based on shard orientation) so that the program and kernels will not be launched
    on cores with no data on them (this can cause failures).
    **/
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Tensor must be on device to compute optimal worker cores.");
    TT_FATAL(tensor.is_sharded(), "Tensor must be sharded to compute optimal worker cores.");
    if (!tensor.memory_config().is_dram()) {
        return tensor.buffer()->buffer_distribution_spec().value().cores_with_data();
    }
    TT_FATAL(tensor.device() != nullptr, "Device pointer must be valid when selecting optimal DRAM worker cores");
    auto all_dram_workers = tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(noc);
    const auto dram_banks = tensor.buffer()->buffer_distribution_spec().value().cores_with_data();
    std::vector<CoreCoord> ordered_worker_cores_with_data;
    ordered_worker_cores_with_data.reserve(dram_banks.size());
    for (const auto& dram_core : dram_banks) {
        const uint32_t dram_channel = tensor.device()->dram_channel_from_logical_core(dram_core);
        ordered_worker_cores_with_data.push_back(all_dram_workers[dram_channel]);
    }
    return ordered_worker_cores_with_data;
}

}  // namespace ttnn
