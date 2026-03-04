// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_view.hpp"

#include <tt_stl/assert.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/mesh_buffer.hpp>

namespace ttnn::operations::experimental::deepseek {

ttnn::Tensor batch_view(const ttnn::Tensor& input_tensor, uint32_t batch_index) {
    // Validate input is on device
    TT_FATAL(
        input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "batch_view: Input tensor must be on device, got {}",
        input_tensor.storage_type());

    // Validate input is not sharded (must be interleaved)
    TT_FATAL(
        !input_tensor.memory_config().is_sharded(),
        "batch_view: Input tensor must be DRAM interleaved, not sharded");

    // Validate input is 3D
    const auto& logical_shape = input_tensor.logical_shape();
    TT_FATAL(
        logical_shape.rank() == 3,
        "batch_view: Input tensor must be 3D [b, M, N], got rank {}",
        logical_shape.rank());

    const uint32_t b = logical_shape[0];
    const uint32_t M = logical_shape[1];
    const uint32_t N = logical_shape[2];

    // Validate batch index
    TT_FATAL(batch_index < b, "batch_view: batch_index {} out of range [0, {})", batch_index, b);

    // Get buffer info
    auto device_storage = input_tensor.device_storage();
    auto* buffer = device_storage.get_buffer();
    const auto page_size = buffer->page_size();
    const auto element_size = input_tensor.element_size();

    // Compute batch dimensions in bytes
    const auto batch_elements = M * N;
    const auto batch_size_bytes = batch_elements * element_size;
    const auto offset_bytes = batch_index * batch_size_bytes;

    // Validate page alignment
    TT_FATAL(
        offset_bytes % page_size == 0,
        "batch_view: Batch offset {} bytes is not page-aligned (page_size={}). "
        "For TILE layout, M*N ({}) must be divisible by {} elements.",
        offset_bytes,
        page_size,
        batch_elements,
        page_size / element_size);

    TT_FATAL(
        batch_size_bytes % page_size == 0,
        "batch_view: Batch size {} bytes is not page-aligned (page_size={}).",
        batch_size_bytes,
        page_size);

    // Create new shape [M, N]
    tt::tt_metal::Shape new_logical_shape({M, N});

    // Create TensorSpec for the new shape
    auto new_spec = tt::tt_metal::TensorSpec(
        new_logical_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            input_tensor.memory_config(),
            new_logical_shape,
            new_logical_shape));

    // Create view MeshBuffer using proper Buffer::view() mechanism
    // This correctly handles interleaved buffer page offsets via root_buffer_offset_
    auto original_mesh_buffer = device_storage.mesh_buffer;

    // Create the view with the correct region (offset and size in bytes)
    tt::tt_metal::BufferRegion region{offset_bytes, batch_size_bytes};
    auto view_mesh_buffer = original_mesh_buffer->view(region);

    // Create DeviceStorage with the view buffer
    // The view MeshBuffer internally holds a reference to the original for lifetime management
    tt::tt_metal::DeviceStorage view_storage(view_mesh_buffer, device_storage.coords, nullptr);

    return ttnn::Tensor(std::move(view_storage), new_spec, input_tensor.tensor_topology());
}

}  // namespace ttnn::operations::experimental::deepseek
