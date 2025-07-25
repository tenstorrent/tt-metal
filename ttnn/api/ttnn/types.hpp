// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/buffer_types.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {
namespace types {

using IDevice = tt::tt_metal::IDevice;
using Program = tt::tt_metal::Program;

constexpr auto TILE_SIZE = 32;

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::ShardMode;
using tt::tt_metal::ShardOrientation;
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

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

struct CoreGrid {
    std::size_t x;
    std::size_t y;

    CoreGrid(std::size_t x, std::size_t y) : x(x), y(y) {}
    CoreCoord to_CoreCoord() { return CoreCoord(int(x), int(y)); }
};

using Buffer = tt::tt_metal::Buffer;

inline std::ostream& operator<<(std::ostream& os, const CoreGrid& core_grid) {
    os << "ttnn.CoreGrid(x=" << core_grid.x << ", y=" << core_grid.y << ")";
    return os;
}

using tt::tt_metal::GlobalSemaphore;
using tt::tt_metal::SubDevice;
using tt::tt_metal::SubDeviceManagerId;
using tt::tt_metal::experimental::GlobalCircularBuffer;

}  // namespace types

using namespace types;

}  // namespace ttnn
