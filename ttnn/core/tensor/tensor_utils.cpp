// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"

#include <tt_stl/overloaded.hpp>

#include "ttnn/tensor/types.hpp"

#include <tracy/Tracy.hpp>

namespace tt::tt_metal {

bool is_cpu_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::HOST; }

bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

CBDescriptor cb_descriptor_from_sharded_tensor(uint8_t cb_index, const Tensor& tensor) {
    TT_FATAL(tensor.is_sharded(), "Tensor must be sharded to automatically create a CBDescriptor");

    return CBDescriptor{
        .total_size = tensor.buffer()->aligned_size_per_bank(),
        .core_ranges = tensor.shard_spec()->grid,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = cb_index,
            .data_format = datatype_to_dataformat_converter(tensor.tensor_spec().tensor_layout().get_data_type()),
            .page_size = tensor.buffer()->aligned_page_size(),
            .tile = TileDescriptor(tensor.tensor_spec().tile())}},
        .buffer = tensor.buffer(),
        .global_circular_buffer = nullptr};
}

}  // namespace tt::tt_metal
