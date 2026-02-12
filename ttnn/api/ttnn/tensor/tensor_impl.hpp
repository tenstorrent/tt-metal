// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace tt::tt_metal::tensor_impl {

// Empty structs to facilitate Tensor template logic.
struct bfloat4_b {};
struct bfloat8_b {};

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

// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec);

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec);

// ======================================================================================
//                                         .to_host() and .to_device()
// ======================================================================================

Tensor to_host(const Tensor& tensor, bool blocking = true, std::optional<QueueId> cq_id = std::nullopt);

void copy_to_host(
    const Tensor& device_tensor,
    Tensor& host_tensor,
    bool blocking = true,
    std::optional<QueueId> cq_id = std::nullopt);

Tensor to_device(
    const Tensor& tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt,
    std::optional<QueueId> cq_id = std::nullopt);

void copy_to_device(const Tensor& host_tensor, Tensor& device_tensor, std::optional<QueueId> cq_id = std::nullopt);

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

Tensor to_layout(const Tensor& tensor, Layout target_layout);

Tensor to_layout_bfloat(const Tensor& tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
Tensor pad(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value);

Tensor unpad(
    const Tensor& tensor, const tt::tt_metal::Shape& output_tensor_start, const tt::tt_metal::Shape& output_tensor_end);

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

Tensor to_dtype(const Tensor& input_tensor, DataType dtype);

// Utility to convert runtime DataType to compile-time constant and dispatch the function call
template <typename Func, typename... Args>
auto dispatch(DataType dtype, Func&& func, Args&&... args) {
    switch (dtype) {
        case DataType::BFLOAT16:
            return (std::forward<Func>(func)).template operator()<bfloat16>(std::forward<Args>(args)...);
        case DataType::FLOAT32:
            return (std::forward<Func>(func)).template operator()<float>(std::forward<Args>(args)...);
        case DataType::INT32:
            return (std::forward<Func>(func)).template operator()<int32_t>(std::forward<Args>(args)...);
        case DataType::UINT32:
            return (std::forward<Func>(func)).template operator()<uint32_t>(std::forward<Args>(args)...);
        case DataType::UINT16:
            return (std::forward<Func>(func)).template operator()<uint16_t>(std::forward<Args>(args)...);
        case DataType::UINT8:
            return (std::forward<Func>(func)).template operator()<uint8_t>(std::forward<Args>(args)...);
        case DataType::BFLOAT8_B:
            return (std::forward<Func>(func)).template operator()<bfloat8_b>(std::forward<Args>(args)...);
        case DataType::BFLOAT4_B:
            return (std::forward<Func>(func)).template operator()<bfloat4_b>(std::forward<Args>(args)...);
        default: TT_THROW("Unsupported data type");
    }
}

}  // namespace tt::tt_metal::tensor_impl
