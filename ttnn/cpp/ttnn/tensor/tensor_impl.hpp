// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include "common/bfloat4.hpp"
#include "common/bfloat8.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_stl/concepts.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

std::array<uint32_t, 2> get_sharded_page_shape(Layout layout, DataType dtype, std::array<uint32_t, 2> shard_shape, const std::optional<Tile>& tile);

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              Low Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                        Data type converters, packers, and unpackers
// ======================================================================================
// TODO(arakhmati): Should cast_vec be a generator?

template <typename OutputDataType, template <typename> typename BufferType, typename InputDataType>
std::vector<OutputDataType> cast_vec(const BufferType<InputDataType>& data_to_convert) {
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

// TODO(arakhmati): Should pack_vec_into_uint32_vec be a generator?
template <typename DataType, template <typename> typename BufferType>
std::vector<uint32_t> pack_vec_into_uint32_vec(const BufferType<DataType>& data_to_pack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return std::vector(std::begin(data_to_pack), std::end(data_to_pack));
    } else if constexpr (std::is_same_v<DataType, int32_t>) {
        std::vector<uint32_t> uint32_data;
        union int32_uint32_convert {
            uint32_t u;
            int32_t i;
            int32_uint32_convert() : u(0) {}
        };
        for (auto i = 0; i < data_to_pack.size(); i++) {
            int32_uint32_convert a;
            a.i = data_to_pack[i];
            uint32_data.push_back(a.u);
        }
        return uint32_data;
    } else if constexpr (std::is_same_v<DataType, uint8_t>) {
        std::vector<uint32_t> output;
        for (auto index = 0; index < data_to_pack.size(); index += 4) {
            auto value = data_to_pack[index + 3] << 24 | data_to_pack[index + 2] << 16 | data_to_pack[index + 1] << 8 | data_to_pack[index];
            output.push_back(value);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, uint16_t>) {
        std::vector<uint32_t> output;
        for (auto index = 0; index < data_to_pack.size(); index += 2) {
            auto value = data_to_pack[index + 1] << 16 | data_to_pack[index];
            output.push_back(value);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, bfloat16>) {
        auto bfloat16_vec = std::vector(std::begin(data_to_pack), std::end(data_to_pack));
        return pack_bfloat16_vec_into_uint32_vec(bfloat16_vec);
    } else if constexpr (std::is_same_v<DataType, float>) {
        std::vector<uint32_t> uint32_data;
        union float_uint32_convert {
            uint32_t u;
            float f;
            float_uint32_convert() : u(0) {}
        };
        for (auto i = 0; i < data_to_pack.size(); i++) {
            float_uint32_convert a;
            a.f = data_to_pack[i];
            uint32_data.push_back(a.u);
        }
        return uint32_data;
    } else {
        static_assert(tt::stl::concepts::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
    }
}

template <typename DataType>
std::vector<DataType> unpack_uint32_vec(std::vector<uint32_t>& data_to_unpack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return data_to_unpack;
    } else if constexpr (std::is_same_v<DataType, int32_t>) {
        union int32_uint32_convert {
            uint32_t u;
            int32_t i;
            int32_uint32_convert() : u(0) {}
        };
        std::vector<int32_t> int32_data;
        for (auto i = 0; i < data_to_unpack.size(); i++) {
            int32_uint32_convert a;
            a.u = data_to_unpack[i];
            int32_data.push_back(a.i);
        }
        return int32_data;
    } else if constexpr (std::is_same_v<DataType, uint8_t>) {
        std::vector<DataType> output;
        for (auto index = 0; index < data_to_unpack.size(); index++) {
            output.push_back((data_to_unpack[index]) & 0xFF);
            output.push_back((data_to_unpack[index] >> 8) & 0xFF);
            output.push_back((data_to_unpack[index] >> 16) & 0xFF);
            output.push_back((data_to_unpack[index] >> 24) & 0xFF);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, uint16_t>) {
        std::vector<DataType> output;
        for (auto index = 0; index < data_to_unpack.size(); index++) {
            output.push_back(data_to_unpack[index] & 0xFFFF);
            output.push_back(data_to_unpack[index] >> 16);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, bfloat16>) {
        return unpack_uint32_vec_into_bfloat16_vec(data_to_unpack);
    } else if constexpr (std::is_same_v<DataType, float>) {
        union float_uint32_convert {
            uint32_t u;
            float f;
            float_uint32_convert() : u(0) {}
        };
        std::vector<float> float_data;
        for (auto i = 0; i < data_to_unpack.size(); i++) {
            float_uint32_convert a;
            a.u = data_to_unpack[i];
            float_data.push_back(a.f);
        }
        return float_data;
    } else {
        static_assert(tt::stl::concepts::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
    }
}

uint32_t element_size_bytes(DataType dtype);

template <typename T>
constexpr inline size_t packed_buffer_size_bytes(size_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(T);
    return (volume_unpacked_data / num_type_in_u32) * sizeof(uint32_t);
}

// Specialization for float because it gets converted to bfloat16 before being packed
template <>
constexpr inline size_t packed_buffer_size_bytes<float>(size_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(float);
    return (volume_unpacked_data / num_type_in_u32) * sizeof(uint32_t);
}

template <>
constexpr inline size_t packed_buffer_size_bytes<bfloat8_b>(size_t volume_unpacked_data) {
    return packed_buffer_size_bytes<uint32_t>(volume_unpacked_data);
}

template <>
constexpr inline size_t packed_buffer_size_bytes<bfloat4_b>(size_t volume_unpacked_data) {
    return packed_buffer_size_bytes<uint32_t>(volume_unpacked_data);
}

// ======================================================================================
//                                  Layout converters
// ======================================================================================
namespace detail {
static std::vector<uint32_t> to_4D_shape(const tt::tt_metal::LegacyShape& shape) {
    if (shape.rank() == 1) {
        return {1, 1, 1, shape[-1]};
    } else if (shape.rank() == 2) {
        return {1, 1, shape[-2], shape[-1]};
    } else if (shape.rank() == 3) {
        return {1, shape[-3], shape[-2], shape[-1]};
    } else if (shape.rank() == 4) {
        return {shape[-4], shape[-3], shape[-2], shape[-1]};
    } else {
        TT_THROW("Rank {} is not supported!", shape.rank());
    }
}

static std::vector<uint32_t> to_vector(const tt::tt_metal::LegacyShape& shape) {
    std::vector<uint32_t> shape_vec;
    for (int i = 0; i < shape.rank(); i++) {
        shape_vec.push_back(shape[i]);
    }
    return shape_vec;
}

}  // namespace detail

template <typename T, template <typename> typename BufferType>
inline std::vector<T> convert_layout_row_major_to_tile(const tt::tt_metal::LegacyShape& shape, const Tile& tile, const BufferType<T>& data_to_convert) {
    TT_FATAL(
        (shape[-2] % tile.get_tile_shape()[0] == 0 && shape[-1] % tile.get_tile_shape()[1] == 0),
        "Unsupported shape for tensor conversion from row-major to tile layout. The tensor shape height and width must be a multiple of tile height ({}) and width ({}), but the provided shape is {}", tile.get_tile_shape()[0], tile.get_tile_shape()[1], shape);

    auto tile_shape = std::vector<uint32_t>{ tile.get_tile_shape()[0], tile.get_tile_shape()[1] };
    auto face_shape = std::vector<uint32_t>{ tile.get_face_shape()[0], tile.get_face_shape()[1] };
    return convert_layout(
        data_to_convert, detail::to_vector(shape), TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED_NFACES, tile_shape, face_shape);
}

template <typename T, template <typename> typename BufferType>
inline std::vector<T> convert_layout_tile_to_row_major(const tt::tt_metal::LegacyShape& shape, const Tile& tile, const BufferType<T>& data_to_convert) {
    auto tile_shape = std::vector<uint32_t>{ tile.get_tile_shape()[0], tile.get_tile_shape()[1] };
    auto face_shape = std::vector<uint32_t>{ tile.get_face_shape()[0], tile.get_face_shape()[1] };
    return convert_layout(
        data_to_convert, detail::to_vector(shape), TensorLayout::TILED_NFACES, TensorLayout::LIN_ROW_MAJOR, tile_shape, face_shape);
}

// ======================================================================================
//                                      Validators
// ======================================================================================
void validate_on_device_dtype_and_layout(Device* device, const ttnn::SimpleShape& shape, DataType dtype, Layout layout);
void validate_sharded_buffer_allocation(
    const ttnn::SimpleShape& shape,
    Layout layout,
    DataType data_type,
    const ShardSpecBuffer& shard_params,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile = std::nullopt);
// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

uint32_t get_page_size(DataType dtype, Layout layout, uint32_t total_size_bytes, const ttnn::SimpleShape& shape, const std::optional<Tile>& tile = std::nullopt);

DeviceBuffer allocate_buffer_on_device(
    Device* device,
    const ttnn::SimpleShape& shape,
    const TensorLayout& layout);

DeviceBuffer allocate_buffer_on_device(
    size_t buffer_size_bytes,
    Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    const std::optional<ShardSpecBuffer>& shard_spec = std::nullopt,
    const std::optional<Tile>& tile = std::nullopt);

template <typename T>
inline void read_data_from_device_buffer(
    CommandQueue& cq, DeviceBuffer device_buffer, void* host_buffer_data, bool blocking) {
    EnqueueReadBuffer(cq, device_buffer, host_buffer_data, blocking);
}

template <typename T>
inline void read_data_from_device_buffer(DeviceBuffer device_buffer, vector<T>& host_buffer) {
    std::vector<uint32_t> host_buffer_uint32;
    ::detail::ReadFromBuffer(device_buffer, host_buffer_uint32);
    host_buffer = unpack_uint32_vec<T>(host_buffer_uint32);
}

// ======================================================================================
//                                         .to()
// ======================================================================================

template <typename T>
Tensor to_host(const Tensor& tensor, bool blocking = true, uint8_t cq_id = ttnn::DefaultQueueId);

template <typename T>
Tensor to_host_sharded(const Tensor& tensor);

template <typename T>
Tensor to_device(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue);

template <typename T>
Tensor to_layout(const Tensor& tensor, Layout target_layout);

template <typename T>
Tensor to_layout_bfloat(const Tensor& tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
template <typename T>
Tensor pad(const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const tt::tt_metal::LegacyShape& input_tensor_start, float pad_value);

template <typename T>
Tensor unpad(const Tensor& tensor, const tt::tt_metal::LegacyShape& output_tensor_start, const tt::tt_metal::LegacyShape& output_tensor_end);

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
std::string to_string(const Tensor& tensor, std::optional<DataType> original_dtype = std::nullopt);

template <typename T>
Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id);

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
