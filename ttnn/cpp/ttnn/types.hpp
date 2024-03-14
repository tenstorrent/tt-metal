// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/types.hpp"

namespace ttnn {
namespace types {

using Device = tt::tt_metal::Device;

constexpr auto TILE_SIZE = 32;

using tt::tt_metal::DataType;
static constexpr auto uint16 = DataType::UINT16;
static constexpr auto uint32 = DataType::UINT32;
static constexpr auto float32 = DataType::FLOAT32;
static constexpr auto bfloat16 = DataType::BFLOAT16;
static constexpr auto bfloat8_b = DataType::BFLOAT8_B;
static constexpr auto bfloat4_b = DataType::BFLOAT4_B;

using tt::tt_metal::BufferType;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::TensorMemoryLayout;

static const auto DRAM_MEMORY_CONFIG = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
static const auto L1_MEMORY_CONFIG = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1};
static const auto L1_BLOCK_SHARDED_MEMORY_CONFIG = MemoryConfig{TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1};
static const auto L1_HEIGHT_SHARDED_MEMORY_CONFIG = MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1};
static const auto L1_WIDTH_SHARDED_MEMORY_CONFIG = MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1};

using tt::tt_metal::Layout;
static constexpr auto ROW_MAJOR_LAYOUT = Layout::ROW_MAJOR;
static constexpr auto TILE_LAYOUT = Layout::TILE;

using tt::tt_metal::StorageType;
static constexpr auto DEVICE_STORAGE_TYPE = StorageType::DEVICE;

struct TensorSchema {
    const std::size_t min_rank;
    const std::size_t max_rank;
    const std::set<DataType> dtypes;
    const std::set<Layout> layouts;
    const bool can_be_on_device;
    const bool can_be_on_cpu;
    const bool can_be_a_scalar;
    const bool is_optional;
};

struct CoreGrid {
    std::size_t x;
    std::size_t y;

    CoreGrid(std::size_t x, std::size_t y) : x(x), y(y) {}
};

static std::ostream &operator<<(std::ostream &os, const CoreGrid &core_grid) {
    os << "ttnn.CoreGrid(x=" <<core_grid.x<<", y="<<core_grid.y<<")";
    return os;
}

}

using namespace types;

}  // namespace ttnn
