// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <optional>

// forward declarations
// #include "tensor/types.hpp" // DataType, Lauout, StorageType
// #include "tt_metal/impl/buffers/buffer_constants.hpp" // TensorMemoryLayout, ShardOrientation
// #include "tt_metal/impl/buffers/buffer.hpp" // BufferType
namespace tt
{
namespace tt_metal
{
enum class DataType;            // tensor/types.hpp
enum class Layout;              // tensor/types.hpp
enum class StorageType;         // tensor/types.hpp
enum class TensorMemoryLayout;  // tt_metal/impl/buffers/buffer_constants.hpp
enum class ShardOrientation;    // tt_metal/impl/buffers/buffer_constants.hpp
enum class BufferType;          // tt_metal/impl/buffers/buffer.hpp
} // namespace tt_metal
} // namespace tt

namespace ttnn {
    namespace types {
        struct Shape;
    }
} // namespace ttnn

namespace ttnn::str_wrapper
{
    std::optional<tt::tt_metal::StorageType> str_to_storage_type(const std::string& storage_type_str);
    std::optional<tt::tt_metal::Layout> str_to_layout(const std::string& layout_str);
    std::optional<tt::tt_metal::TensorMemoryLayout> str_to_memory_layout(const std::string& tensor_memory_layout_str);
    std::optional<tt::tt_metal::DataType> str_to_data_type(const std::string& data_type_str);
    std::optional<tt::tt_metal::BufferType> str_to_buffer_type(const std::string& buffer_type_str);
    std::optional<tt::tt_metal::ShardOrientation> str_to_shard_orientation(const std::string& shard_str);
} // namespace ttnn::wrapper
