// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "types_wrapper.hpp"
#include <optional>

#include "tensor/types.hpp" // DataType, Layout, StorageType, MemoryConfig
#include "tt_metal/impl/buffers/buffer_constants.hpp" // TensorMemoryLayout, ShardOrientation
#include "tt_metal/impl/buffers/buffer.hpp" // BufferType, ShardSpec
#include "common/core_coord.h" // CoreRangeSet

namespace ttnn::str_wrapper
{

std::optional<tt::tt_metal::StorageType> to_storage_type(const std::string& storage_type_str)
{
    if (storage_type_str == "OWNED") return tt::tt_metal::StorageType::OWNED;
    if (storage_type_str == "DEVICE") return tt::tt_metal::StorageType::DEVICE;
    if (storage_type_str == "BORROWED") return tt::tt_metal::StorageType::BORROWED;
    if (storage_type_str == "MULTI_DEVICE") return tt::tt_metal::StorageType::MULTI_DEVICE;
    if (storage_type_str == "MULTI_DEVICE_HOST") return tt::tt_metal::StorageType::MULTI_DEVICE_HOST;
    return std::nullopt;
}

std::optional<tt::tt_metal::Layout> to_layout(const std::string& layout_str)
{
    if (layout_str == "ROW_MAJOR") return tt::tt_metal::Layout::ROW_MAJOR;
    if (layout_str == "TILE") return tt::tt_metal::Layout::TILE;
    if (layout_str == "INVALID") return tt::tt_metal::Layout::INVALID;
    return std::nullopt;
}

std::optional<tt::tt_metal::TensorMemoryLayout> to_tensor_memory_layout(const std::string& tensor_memory_layout_str)
{
    if (tensor_memory_layout_str == "INTERLEAVED") return tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    if (tensor_memory_layout_str == "SINGLE_BANK") return tt::tt_metal::TensorMemoryLayout::SINGLE_BANK;
    if (tensor_memory_layout_str == "HEIGHT_SHARDED") return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
    if (tensor_memory_layout_str == "WIDTH_SHARDED") return tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
    if (tensor_memory_layout_str == "BLOCK_SHARDED") return tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
    return std::nullopt;
}

std::optional<tt::tt_metal::DataType> to_data_type(const std::string& data_type_str)
{
    if (data_type_str == "BFLOAT16") return tt::tt_metal::DataType::BFLOAT16;
    if (data_type_str == "FLOAT32") return tt::tt_metal::DataType::FLOAT32;
    if (data_type_str == "UINT32") return tt::tt_metal::DataType::UINT32;
    if (data_type_str == "BFLOAT8_B") return tt::tt_metal::DataType::BFLOAT8_B;
    if (data_type_str == "BFLOAT4_B") return tt::tt_metal::DataType::BFLOAT4_B;
    if (data_type_str == "UINT8") return tt::tt_metal::DataType::UINT8;
    if (data_type_str == "UINT16") return tt::tt_metal::DataType::UINT16;
    if (data_type_str == "INT32") return tt::tt_metal::DataType::INT32;
    if (data_type_str == "INVALID") return tt::tt_metal::DataType::INVALID;
    return std::nullopt;
}

std::optional<tt::tt_metal::BufferType> to_buffer_type(const std::string& buffer_type_str)
{
    if (buffer_type_str == "DRAM") return tt::tt_metal::BufferType::DRAM;
    if (buffer_type_str == "L1") return tt::tt_metal::BufferType::L1;
    if (buffer_type_str == "SYSTEM_MEMORY") return tt::tt_metal::BufferType::SYSTEM_MEMORY;
    if (buffer_type_str == "L1_SMALL") return tt::tt_metal::BufferType::L1_SMALL;
    if (buffer_type_str == "TRACE") return tt::tt_metal::BufferType::TRACE;
    return std::nullopt;
}

std::optional<tt::tt_metal::ShardOrientation> to_shard_orientation(const std::string& shard_str)
{
    if (shard_str == "ROW_MAJOR") return tt::tt_metal::ShardOrientation::ROW_MAJOR;
    if (shard_str == "COL_MAJOR") return tt::tt_metal::ShardOrientation::COL_MAJOR;
    return std::nullopt;
}

// std::optional<ttnn::Shape> vector_to_shard_shape(const std::vector<uint32_t>& shard_shape_vector)
// {
//     return ttnn::Shape(shard_shape_vector);
// }

// Layout layout_by_index(const int index, const std::vector<std::string>& layouts_str)
// {
//     return to_layout(layouts_str.at(index)).value();
// }

// DataType datatype_by_index(const int index, const std::vector<std::string>& data_types_str)
// {
//     return to_data_type(data_types_str.at(index)).value();
// }

// tt::tt_metal::Shape shape_by_index(const int index, const std::vector<std::vector<uint32_t>>& shapes_vectors)
// {
//     return vector_to_shape(shapes_vectors.at(index));
// }

// bool is_sharded_by_index(const int index, const std::vector<bool>& shareded_vector)
// {
//     return shareded_vector.at(0);
// }

// tt::tt_metal::TensorMemoryLayout memory_layout_by_index(const int index, const std::vector<std::string>& memory_layouts)
// {
//     return to_memory_layout(memory_layouts.at(index)).value();
// }

// ShardOrientation shard_orientation_by_index(const int index, const std::vector<std::string>& shards_str)
// {
//     return to_shard_orientation(shards_str.at(index)).value();
// }

// tt::tt_metal::BufferType buffer_type_by_index(const int index, const std::vector<std::string>& buffer_types_str)
// {
//     return to_buffer_type(buffer_types_str.at(index)).value();
// }

// const uint32_t volume(tt::tt_metal::Shape& shape)
// {
//     auto rank = shape.rank();
//     auto volume = 1;
//     for (auto index = 0; index < rank; index++) {
//         volume *= shape.operator[](index);
//     }
//     return volume;
// }

// ttnn::Shape shard_shape_by_index(const int index, const std::vector<std::vector<uint32_t>>& shard_shapes)
// {
//     return vector_to_shard_shape(shard_shapes.at(index)).value();
// }

// CoreRangeSet get_core_range_set_by_index(const int index, const std::vector<CoreRangeSet>& core_range_set)
// {
//     return core_range_set.at(index);
// }

} // namespace ttnn::str_wrapper

namespace ttnn::vector_wrapper
{
    tt::tt_metal::Shape to_shape(const std::vector<uint32_t>& shape_vector)
    {
        return tt::tt_metal::Shape(shape_vector);
    }
} // namespace ttnn::vector_wrapper

namespace ttnn::tuple_wrapper {
    std::optional<tt::tt_metal::ShardSpec> to_shard_spec(
        const std::vector<std::array<uint32_t, 4>>& core_range_set,
        const std::array<uint32_t, 2>& shard_shape,
        const std::string& shard_orientation,
        const bool& halo = false) {

        if (core_range_set.size() == 0)
        {
            return std::nullopt;
        }

        std::set<CoreRange> core_ranges;
        for (const auto& core_range : core_range_set)
        {
            core_ranges.insert(CoreRange{CoreCoord{core_range[0], core_range[1]}, CoreCoord{core_range[2], core_range[3]}});
        }

        return std::make_optional(tt::tt_metal::ShardSpec{
            CoreRangeSet{core_ranges},
            shard_shape,
            str_wrapper::to_shard_orientation(shard_orientation).value(),
            halo
        });
    }

    std::optional<tt::tt_metal::ShardSpec> to_shard_spec(const mlir_interface::shard_spec_tuple& shard_spec_tuple) {
        return to_shard_spec(
            std::get<std::vector<std::array<uint32_t, 4>>>(shard_spec_tuple),
            std::get<std::array<uint32_t, 2>>(shard_spec_tuple),
            std::get<std::string>(shard_spec_tuple),
            std::get<bool>(shard_spec_tuple));
    }

    std::optional<tt::tt_metal::MemoryConfig> to_memory_config(
        const std::string& tensor_memory_layout,
        const std::string& buffer_type,
        const std::optional<mlir_interface::shard_spec_tuple>& shard_spec) {

        auto deserialized_tensor_memory_layout = str_wrapper::to_tensor_memory_layout(tensor_memory_layout);
        if (!deserialized_tensor_memory_layout.has_value()) {
            return std::nullopt;
        }

        auto deserialized_buffer_type = str_wrapper::to_buffer_type(buffer_type);
        if (!deserialized_buffer_type.has_value()) {
            return std::nullopt;
        }

        auto deserialized_shard_spec = shard_spec.has_value() ? to_shard_spec(shard_spec.value()) : std::nullopt;
        return std::make_optional(tt::tt_metal::MemoryConfig{
            deserialized_tensor_memory_layout.value(),
            deserialized_buffer_type.value(),
            deserialized_shard_spec
        });
    }

    std::optional<tt::tt_metal::MemoryConfig> to_memory_config(const mlir_interface::memory_config_tuple& memory_config_tuple) {
        return to_memory_config(
            std::get<0>(memory_config_tuple),
            std::get<1>(memory_config_tuple),
            std::get<std::optional<mlir_interface::shard_spec_tuple>>(memory_config_tuple));
    }

} // namespace ttnn::tuple_wrapper
