// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::experimental {

ttnn::Tensor view(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    return tt::tt_metal::view(tensor, logical_shape, padded_shape);
}

ttnn::Tensor view(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tt::tt_metal::view(tensor, shape, shape);
}

ttnn::Tensor view(const ttnn::Tensor& tensor, ttsl::Span<const int32_t> shape_vector) {
    auto shape = ttnn::operations::data_movement::detail::infer_dims_for_reshape(tensor, shape_vector);
    return tt::tt_metal::view(tensor, shape, shape);
}

ttnn::Tensor view(const ttnn::Tensor& tensor, const ttsl::SmallVector<int32_t>& shape_vector) {
    return ttnn::experimental::view(tensor, ttsl::Span<const int32_t>(shape_vector.data(), shape_vector.size()));
}

ttnn::Tensor alias_view(const ttnn::Tensor& tensor, const ttnn::Shape& shape, tt::tt_metal::DataType dtype) {
    using namespace tt::tt_metal;
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.is_allocated(),
        "alias_view: input must be an allocated device tensor");
    TT_FATAL(tensor.layout() == Layout::TILE, "alias_view: input must be TILE layout, got {}", tensor.layout());
    const MemoryConfig& memory_config = tensor.memory_config();
    TT_FATAL(
        memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "alias_view: input must be interleaved, got {}",
        memory_config.memory_layout());
    TT_FATAL(
        tensor.padded_shape().rank() >= 3,
        "alias_view: input rank must be at least 3 so that dim 0 is outside the tiled dims, got {}",
        tensor.padded_shape().rank());

    const TensorSpec view_spec(
        shape, TensorLayout(dtype, PageConfig(Layout::TILE, tensor.tensor_spec().tile()), memory_config));
    const auto& owner = tensor.device_storage().get_mesh_buffer();
    const uint32_t banks = owner.device()->allocator()->get_num_banks(memory_config.buffer_type());
    const DeviceAddr owner_page = owner.aligned_page_size();
    const DeviceAddr view_page = view_spec.compute_page_size_bytes();
    const DeviceAddr view_aligned_page = tt::round_up(view_page, owner.alignment());

    // Interleaved page p sits on bank p % banks at offset (p / banks) * aligned page size. A dim-0 slice of the input
    // (one block of a paged cache) owns the same byte range on every bank only when its page count is a multiple of
    // the bank count. When that range also holds a whole number of view pages, view pages
    // [i * per_slice, (i + 1) * per_slice) cover exactly slice i, with per_slice = banks * range / view page. A dim-0
    // slice of the view must then nest inside one input slice or cover whole input slices.
    const uint32_t slices = tensor.padded_shape()[0];
    const uint64_t pages_per_slice = owner.num_pages() / slices;
    TT_FATAL(
        pages_per_slice % banks == 0,
        "alias_view: a dim-0 slice of the input has {} pages, not a multiple of the {} banks, so slices of the input "
        "and of the view would not cover the same bytes",
        pages_per_slice,
        banks);
    const DeviceAddr slice_bytes_per_bank = pages_per_slice / banks * owner_page;
    TT_FATAL(
        slice_bytes_per_bank % view_aligned_page == 0,
        "alias_view: a dim-0 slice of the input holds {} bytes per bank, not a multiple of the view's {} byte page",
        slice_bytes_per_bank,
        view_aligned_page);

    TT_FATAL(shape.rank() >= 3, "alias_view: view rank must be at least 3, got {}", shape.rank());
    const uint64_t view_pages = view_spec.compute_packed_buffer_size_bytes() / view_page;
    const uint64_t per_input_slice = banks * (slice_bytes_per_bank / view_aligned_page);
    const uint64_t per_view_slice = view_pages / view_spec.padded_shape()[0];
    TT_FATAL(
        per_input_slice % per_view_slice == 0 || per_view_slice % per_input_slice == 0,
        "alias_view: a dim-0 slice of the view ({} pages) would straddle dim-0 slices of the input ({} view pages "
        "each)",
        per_view_slice,
        per_input_slice);
    const DeviceAddr view_bytes_per_bank = tt::div_up(view_pages, banks) * view_aligned_page;
    TT_FATAL(
        view_bytes_per_bank <= owner.aligned_size_per_bank(),
        "alias_view: the view needs {} bytes per bank, the input owns {}",
        view_bytes_per_bank,
        owner.aligned_size_per_bank());

    auto device_config = owner.device_local_config();
    device_config.page_size = view_page;
    namespace distributed = tt::tt_metal::distributed;
    const distributed::ReplicatedBufferConfig global_config{.size = view_spec.compute_packed_buffer_size_bytes()};
    auto view_buffer = distributed::MeshBuffer::create(global_config, device_config, owner.device(), owner.address());
    MeshTensor view_mesh_tensor =
        mesh_tensor_from_buffer_with_topology(std::move(*view_buffer), view_spec, tensor.tensor_topology());
    DeviceStorage view_storage(tensor.device_storage(), std::move(view_mesh_tensor));
    return ttnn::Tensor(std::move(view_storage));
}

ttnn::Tensor view(const ttnn::Tensor& tensor, int32_t N, int32_t C, int32_t H, int32_t W) {
    ttsl::SmallVector<int32_t> shape_vec{N, C, H, W};
    return ttnn::experimental::view(tensor, ttsl::Span<const int32_t>(shape_vec.data(), shape_vec.size()));
}

}  // namespace ttnn::experimental
