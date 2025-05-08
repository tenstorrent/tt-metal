// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include <algorithm>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_impl.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "cpp/ttnn/tensor/enum_types.hpp"

#include "ttnn/tensor/shape/shape.hpp"

namespace tt {

namespace tt_metal {

static constexpr std::uint8_t VERSION_ID = 5;

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3,
    BFLOAT4_B = 4,
    UINT8 = 5,
    UINT16 = 6,
    INT32 = 7,
    INVALID = 8,
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& data_type);

template <typename T>
consteval inline DataType convert_to_data_type() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return DataType::UINT8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return DataType::UINT16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::INT32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UINT32;
    } else if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, ::bfloat16>) {
        return DataType::BFLOAT16;
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported DataType!");
    }
}

bool is_floating_point(DataType dtype);

bool is_block_float(DataType dtype);

// Enums are explicitly enumerated due to serialization dependency
// TODO: #16067 - This shouldn't be needed. Serialize this enum to flatbuffer.
enum class StorageType {
    HOST = 0,
    DEVICE = 1,
    MULTI_DEVICE_HOST = 4,  // host storage for multi-device context
};

tt::DataFormat datatype_to_dataformat_converter(DataType datatype);

static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;

using Array1D = std::array<uint32_t, 1>;
using Array2D = std::array<uint32_t, 2>;
using Array3D = std::array<uint32_t, 3>;
using Array4D = std::array<uint32_t, 4>;
using Array5D = std::array<uint32_t, 5>;
using Array6D = std::array<uint32_t, 6>;
using Array7D = std::array<uint32_t, 7>;
using Array8D = std::array<uint32_t, 8>;

class MemoryConfig final {
public:
    MemoryConfig() = default; // Interleaved DRAM
    explicit MemoryConfig(TensorMemoryLayout memory_layout,
        BufferType buffer_type = BufferType::DRAM,
        std::optional<ShardSpec> shard_spec = std::nullopt) :
        memory_layout_(memory_layout), buffer_type_(buffer_type), shard_spec_(std::move(shard_spec)) {}
    MemoryConfig(const MemoryConfig& other) = default;
    MemoryConfig& operator=(const MemoryConfig& other) = default;
    MemoryConfig(MemoryConfig&& other) noexcept = default;
    MemoryConfig& operator=(MemoryConfig&& other) noexcept = default;

    TensorMemoryLayout memory_layout() const { return memory_layout_; }
    BufferType buffer_type() const { return buffer_type_; }
    const std::optional<ShardSpec>& shard_spec() const { return shard_spec_; }

    MemoryConfig with_shard_spec(std::optional<ShardSpec> shard_spec) const {
        return MemoryConfig(memory_layout_, buffer_type_, std::move(shard_spec));
    }

    bool is_sharded() const;
    bool is_l1() const;
    bool is_dram() const;

    static constexpr auto attribute_names = std::forward_as_tuple("memory_layout", "buffer_type", "shard_spec");
    auto attribute_values() const { return std::forward_as_tuple(memory_layout_, buffer_type_, shard_spec_); }

    friend std::ostream& operator<<(std::ostream& os, const MemoryConfig& config);

private:
    TensorMemoryLayout memory_layout_ = TensorMemoryLayout::INTERLEAVED;  // Interleave the data across multiple banks
    BufferType buffer_type_ = BufferType::DRAM;                           // Can be either DRAM or L1
    std::optional<ShardSpec> shard_spec_ = std::nullopt;
};

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config);

bool operator==(const MemoryConfig &config_a, const MemoryConfig &config_b);
bool operator!=(const MemoryConfig &config_a, const MemoryConfig &config_b);

} // namespace tt_metal
} // namespace tt

template <>
struct tt::stl::json::to_json_t<tt::tt_metal::MemoryConfig> {
    nlohmann::json operator()(const tt::tt_metal::MemoryConfig& config) const;
};

template <>
struct tt::stl::json::from_json_t<tt::tt_metal::MemoryConfig> {
    tt::tt_metal::MemoryConfig operator()(const nlohmann::json& json_object) const;
};
