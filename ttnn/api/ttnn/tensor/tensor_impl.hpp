// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <tracy/Tracy.hpp>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/types.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

// Empty structs to facilitate Tensor template logic.
struct bfloat4_b {};
struct bfloat8_b {};

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              Low Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                        Data type converters, packers, and unpackers
// ======================================================================================

template <typename OutputDataType, typename InputDataType>
std::vector<OutputDataType> cast_vec(tt::stl::Span<const InputDataType> data_to_convert) {
    std::vector<OutputDataType> converted_data;
    for (auto datum : data_to_convert) {
        if constexpr (std::is_same_v<OutputDataType, float> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back(datum.to_float());
        } else if constexpr (std::is_same_v<OutputDataType, uint32_t> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back((uint32_t)datum.to_uint16());
        } else {
            converted_data.push_back(static_cast<OutputDataType>(datum));
        }
    }
    return converted_data;
}

uint32_t element_size_bytes(DataType dtype);

template <typename T>
constexpr size_t packed_buffer_size_bytes(size_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(T);
    return (volume_unpacked_data / num_type_in_u32) * sizeof(uint32_t);
}

// Specialization for float because it gets converted to bfloat16 before being packed
template <>
constexpr size_t packed_buffer_size_bytes<float>(size_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(float);
    return (volume_unpacked_data / num_type_in_u32) * sizeof(uint32_t);
}

template <>
constexpr size_t packed_buffer_size_bytes<bfloat8_b>(size_t volume_unpacked_data) {
    return packed_buffer_size_bytes<uint32_t>(volume_unpacked_data);
}

template <>
constexpr size_t packed_buffer_size_bytes<bfloat4_b>(size_t volume_unpacked_data) {
    return packed_buffer_size_bytes<uint32_t>(volume_unpacked_data);
}

// ======================================================================================
//                                  Layout converters
// ======================================================================================
template <typename T>
std::vector<T> convert_layout_row_major_to_tile(
    const Shape2D& shape, const Tile& tile, tt::stl::Span<const T> data_to_convert) {
    if (shape.width() * shape.height() == 0) {
        return std::vector<T>();
    }
    TT_FATAL(
        (shape.height() % tile.get_tile_shape()[0] == 0 && shape.width() % tile.get_tile_shape()[1] == 0),
        "Unsupported shape for tensor conversion from row-major to tile layout. The tensor shape height and width must "
        "be a multiple of tile height ({}) and width ({}), but the provided shape is {}",
        tile.get_tile_shape()[0],
        tile.get_tile_shape()[1],
        shape);

    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    return convert_layout(
        data_to_convert,
        shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        tile_shape,
        face_shape,
        transpose_within_face,
        transpose_of_faces);
}

template <typename T>
std::vector<T> convert_layout_tile_to_row_major(
    const Shape2D& shape, const Tile& tile, tt::stl::Span<const T> data_to_convert) {
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

// Converts logical data into physical data based on tensor spec
// - Logical data: Flat container of row major data corresponding to some ND logical shape
// - Physical data: Flat container of physical data corresponding to tensor spec. It takes into account:
//   * Sharding: Each shard will be padded to nearest page (if needed)
//     ** This is mostly for logical sharding, since logical shards may not be aligned to page in general
//     ** For interleaved, it will be handled as a "logically sharded" tensor with same shard shard height/width
//        as the original tensor dims at -2 and -1. In the future, interleaved may be generalized as sharded.
//     ** This means padding may be inserted in the middle of logical data (if needed)
//   * Layout: Each aligned shard will be tilized (if needed)
//     ** Tilization happens after first inserting padding to align shards (if needed)
//     ** For the last shard, we only align to nearest page instead of full shard size for partial shards
//   * After conversion, size of physical data will match 2D physical size indicated by tensor_spec.physical_shape()
template <typename T>
std::vector<T> encode_tensor_data(tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value = 0);

// Converts physical data into logical data based on tensor spec (see encode_tensor_data for details)
// - Physical data: Flat container of physical data corresponding to tensor spec
//   * Assumes that the physical data already matches tensor spec
//   * There is a bare minimum check that size of physical data matches size indicated by tensor_spec.physical_shape()
// - Logical data: Flat container of row major data corresponding to some ND logical shape
//   * To get logical data, perform the exact inverse process of encode_tensor_data
//   * Resulting data is safe to be converted to python tensors or general consumption with just a ND logical shape
template <typename T>
std::vector<T> decode_tensor_data(tt::stl::Span<const T> physical_data, const TensorSpec& tensor_spec);

// Returns true if the logical tensor data matches the physical tensor data:
// 1. Row major layout is used.
// 2. Logical 2D shape matches physical shape.
// Used for optimizing conversion operations.
bool logical_matches_physical(const TensorSpec& tensor_spec);

// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

std::shared_ptr<Buffer> allocate_buffer_on_device(IDevice* device, const TensorSpec& tensor_spec);

std::shared_ptr<distributed::MeshBuffer> allocate_mesh_buffer_on_device(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec);

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec);

template <typename T>
void read_data_from_device_buffer(CommandQueue& cq, Buffer& device_buffer, void* host_buffer_data, bool blocking) {
    EnqueueReadBuffer(cq, device_buffer, host_buffer_data, blocking);
}

template <typename T>
void read_data_from_device_buffer(Buffer& device_buffer, std::vector<T>& host_buffer) {
    ::tt::tt_metal::detail::ReadFromBuffer(device_buffer, host_buffer);
}

// ======================================================================================
//                                         .to_host() and .to_device()
// ======================================================================================

template <typename T>
Tensor to_host(const Tensor& tensor, bool blocking = true, QueueId cq_id = ttnn::DefaultQueueId);

// TODO: #17215 - This will eventually subsume `to_host`, when "mesh buffer" backed tensors become the default.
template <typename T>
Tensor to_host_mesh_tensor(const Tensor& tensor, bool blocking = true, QueueId cq_id = ttnn::DefaultQueueId);

template <typename T>
void copy_to_host_tensor(
    const Tensor& device_tensor, Tensor& host_tensor, bool blocking = true, QueueId cq_id = ttnn::DefaultQueueId);

template <typename T>
Tensor to_device(
    const Tensor& tensor,
    IDevice* target_device,
    const MemoryConfig& memory_config,
    QueueId cq_id = ttnn::DefaultQueueId);

// TODO: #17215 - This will eventually subsume `to_device`, when "mesh buffer" backed tensors become the default.
template <typename T>
Tensor to_device_mesh_tensor(
    const Tensor& tensor,
    distributed::MeshDevice* mesh_device,
    const MemoryConfig& memory_config,
    QueueId cq_id = ttnn::DefaultQueueId);

template <typename T>
void copy_to_device_tensor(const Tensor& host_tensor, Tensor& device_tensor, QueueId cq_id = ttnn::DefaultQueueId);

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

template <typename T>
Tensor to_layout(const Tensor& tensor, Layout target_layout);

template <typename T>
Tensor to_layout_bfloat(const Tensor& tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
template <typename T>
Tensor pad(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value);

template <typename T>
Tensor unpad(const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);

// ======================================================================================
//                                         Print
// ======================================================================================

std::ostream& operator<<(std::ostream& os, const DataType& dtype);

enum class TensorPrintProfile {
    Empty,
    Short,
    Full,
};

extern TensorPrintProfile TTNN_TENSOR_PRINT_PROFILE;

template <typename T>
std::string to_string(const Tensor& tensor);

template <typename T>
Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id);

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
