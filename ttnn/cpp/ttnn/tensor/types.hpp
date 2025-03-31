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
#include "ttnn/tensor/host_buffer/types.hpp"
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

enum class StorageType {
    OWNED,
    DEVICE,
    BORROWED,           // for storing torch/numpy/etc tensors
    MULTI_DEVICE,       // on-device storage for multi-device context
    MULTI_DEVICE_HOST,  // host storage for multi-device context
};

tt::DataFormat datatype_to_dataformat_converter(DataType datatype);

static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;

typedef std::array<uint32_t, 1> Array1D;
typedef std::array<uint32_t, 2> Array2D;
typedef std::array<uint32_t, 3> Array3D;
typedef std::array<uint32_t, 4> Array4D;
typedef std::array<uint32_t, 5> Array5D;
typedef std::array<uint32_t, 6> Array6D;
typedef std::array<uint32_t, 7> Array7D;
typedef std::array<uint32_t, 8> Array8D;

struct MemoryConfig {
    TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;  // Interleave the data across multiple banks
    BufferType buffer_type = BufferType::DRAM;                           // Can be either DRAM or L1
    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool is_sharded() const;
    bool is_l1() const;
    bool is_dram() const;
};

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config);

bool operator==(const MemoryConfig &config_a, const MemoryConfig &config_b);
bool operator!=(const MemoryConfig &config_a, const MemoryConfig &config_b);

} // namespace tt_metal
} // namespace tt
