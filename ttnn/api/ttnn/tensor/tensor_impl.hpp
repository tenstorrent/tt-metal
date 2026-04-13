// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>
#include <span>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace tt::tt_metal::tensor_impl {

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              Low Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                                  Layout converters
// ======================================================================================

template <typename T>
std::vector<T> convert_layout_tile_to_row_major(
    const Shape2D& shape, const Tile& tile, ttsl::Span<const T> data_to_convert) {
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    return convert_layout(
        data_to_convert,
        shape,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR,
        tile_shape,
        face_shape,
        transpose_within_face,
        transpose_of_faces);
}

// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec);

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec);

MeshTensor allocate_mesh_tensor(
    const TensorSpec& tensor_spec, distributed::MeshDevice& device, TensorTopology topology);

// ======================================================================================
//                   Uniform enqueue_read/write_mesh_tensor
// ======================================================================================

HostTensor enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, bool blocking = true);

void enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, HostTensor& host_tensor, bool blocking = true);

MeshTensor enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt);

void enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor);

// ======================================================================================
//                Non-uniform enqueue_read/write_mesh_tensor
// ======================================================================================

// Data movement for tensors whose shards don't cover the entire MeshDevice.
// The host-side DistributedHostBuffer only populates a subset of MeshCoordinates,
// so the resulting DeviceStorage must track which coordinates were actually written.
namespace non_uniform_data_movement {

HostTensor enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking = true);

void enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    HostTensor& host_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking = true);

std::pair<MeshTensor, std::vector<distributed::MeshCoordinate>> enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt);

std::vector<distributed::MeshCoordinate> enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor);

}  // namespace non_uniform_data_movement

// ======================================================================================
//                    Unit Tensor enqueue_read/write_mesh_tensor
// ======================================================================================

void enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region = std::nullopt);

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

HostTensor to_layout(const HostTensor& tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
HostTensor pad(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value);

HostTensor pad_to_tile(const HostTensor& tensor, float pad_value);

HostTensor unpad(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end);

HostTensor unpad_from_tile(const HostTensor& tensor, const tt::tt_metal::Shape& output_tensor_shape);

// ======================================================================================
//                                  .view()
// ======================================================================================

HostTensor view(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

// ======================================================================================
//                                         Print
// ======================================================================================

std::ostream& operator<<(std::ostream& os, const DataType& dtype);

enum class TensorPrintProfile {
    Empty,
    Short,
    Full,
};

enum class SciMode {
    Enable,
    Disable,
    Default,
};

struct PrintOptions {
    TensorPrintProfile profile = TensorPrintProfile::Short;
    SciMode sci_mode = SciMode::Default;
    int precision = 4;
};

extern PrintOptions TTNN_PRINT_OPTIONS;

std::string to_string(const Tensor& tensor);

Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id);

HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype);

// ======================================================================================
//                                  HostTensor Factory Functions
// ======================================================================================
// These functions create HostTensor objects without device involvement.
// They are the underlying implementation for Tensor::from_xxx methods.

namespace host_tensor {

template <typename T>
HostTensor from_span(ttsl::Span<const T> buffer, const TensorSpec& spec, T pad_value = 0);

template <typename T>
HostTensor from_borrowed_data(
    ttsl::Span<T> buffer, const Shape& shape, MemoryPin pin, const std::optional<Tile>& tile = std::nullopt);

template <typename T>
HostTensor from_vector(const std::vector<T>& buffer, const TensorSpec& spec, T pad_value = 0);

template <typename T>
HostTensor from_vector(std::vector<T>&& buffer, const TensorSpec& spec, T pad_value = 0);

template <typename T>
std::vector<T> to_vector(const HostTensor& tensor);

}  // namespace host_tensor

}  // namespace tt::tt_metal::tensor_impl
