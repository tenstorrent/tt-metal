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

    // Validate input is 3D or 4D
    const auto& logical_shape = input_tensor.logical_shape();
    const auto rank = logical_shape.rank();
    TT_FATAL(
        rank == 3 || rank == 4,
        "batch_view: Input tensor must be 3D [b, M, N] or 4D [X, b, M, N], got rank {}",
        rank);

    // Extract dimensions based on rank
    // 3D: [b, M, N] -> [M, N]
    // 4D: [X, b, M, N] -> [X, M, N] (batch dim is 1)
    uint32_t X = 1;  // Leading dimension for 4D case
    uint32_t b, M, N;

    if (rank == 3) {
        b = logical_shape[0];
        M = logical_shape[1];
        N = logical_shape[2];
    } else {  // rank == 4
        X = logical_shape[0];
        b = logical_shape[1];
        M = logical_shape[2];
        N = logical_shape[3];
    }

    // Validate batch index
    TT_FATAL(batch_index < b, "batch_view: batch_index {} out of range [0, {})", batch_index, b);

    // Get buffer info
    auto device_storage = input_tensor.device_storage();
    auto* buffer = device_storage.get_buffer();
    const auto page_size = buffer->page_size();
    const auto element_size = input_tensor.element_size();

    // Compute batch dimensions in bytes
    // For 4D: each "batch slice" is [X, M, N], but batches are interleaved as [X, b, M, N]
    // So batch i contains elements at positions [x, i, :, :] for all x
    // In memory layout (row-major for logical shape): batch i is NOT contiguous for 4D!
    // However, for the multi-device sharding case, after sharding X across devices,
    // each device has [1, b, M, N] which in memory is effectively [b, M, N]
    // So the batch elements are contiguous: batch i is at offset i * M * N
    const auto offset_bytes = batch_index * M * N * element_size;

    // For 4D tensors with X > 1, batches are NOT contiguous in memory
    // This only works when X == 1 (typical after sharding)
    TT_FATAL(
        X == 1 || rank == 3,
        "batch_view: 4D tensor with first dim > 1 not supported (batches not contiguous). "
        "Got shape [{}, {}, {}, {}]. Use squeeze first or shard so first dim is 1.",
        X,
        b,
        M,
        N);

    // Validate page alignment
    TT_FATAL(
        offset_bytes % page_size == 0,
        "batch_view: Batch offset {} bytes is not page-aligned (page_size={}). "
        "For TILE layout, M*N ({}) must be divisible by {} elements.",
        offset_bytes,
        page_size,
        M * N,
        page_size / element_size);

    const auto slice_size_bytes = M * N * element_size;
    TT_FATAL(
        slice_size_bytes % page_size == 0,
        "batch_view: Batch size {} bytes is not page-aligned (page_size={}).",
        slice_size_bytes,
        page_size);

    // Create new shape: always [M, N] since we only support X=1 for 4D
    // This makes 4D batch_view with X=1 behave identically to 3D batch_view
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
    tt::tt_metal::BufferRegion region{offset_bytes, slice_size_bytes};
    auto view_mesh_buffer = original_mesh_buffer->view(region);

    // Create DeviceStorage with the view buffer
    // The view MeshBuffer internally holds a reference to the original for lifetime management
    tt::tt_metal::DeviceStorage view_storage(view_mesh_buffer, device_storage.coords, nullptr);

    return ttnn::Tensor(std::move(view_storage), new_spec, input_tensor.tensor_topology());
}

}  // namespace ttnn::operations::experimental::deepseek
