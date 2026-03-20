// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"

#include <tt_stl/overloaded.hpp>

#include "ttnn/tensor/types.hpp"

#include <tracy/Tracy.hpp>

namespace tt::tt_metal {

bool logical_matches_physical(const TensorSpec& tensor_spec) {
    return tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape();
}

bool is_cpu_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::HOST; }

bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

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

std::vector<CoreCoord> get_optimal_worker_cores_in_sharded_tensor(const Tensor& tensor, NOC noc) {
    /**
    This function takes in a sharded device tensor (can be legacy 2D sharded or ND sharded) and returns the optimal
    worker cores for the tensor.

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
    TT_FATAL(
        tensor.is_sharded(),
        "Tensor must be sharded to compute optimal worker cores.");  // Host tensors will fail this check.
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

}  // namespace tt::tt_metal
