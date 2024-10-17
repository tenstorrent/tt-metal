// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <mutex>
#include "impl/buffers/buffer.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {
namespace types {

using Device = tt::tt_metal::Device;

constexpr auto TILE_SIZE = 32;

using tt::tt_metal::DataType;
static constexpr auto uint8 = DataType::UINT8;
static constexpr auto uint16 = DataType::UINT16;
static constexpr auto int32 = DataType::INT32;
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
static constexpr auto MULTI_DEVICE_STORAGE_TYPE = StorageType::MULTI_DEVICE;

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

struct CoreGrid {
    std::size_t x;
    std::size_t y;

    CoreGrid(std::size_t x, std::size_t y) : x(x), y(y) {}
    CoreCoord to_CoreCoord(){
        return CoreCoord(int(x), int(y));
    }
};

class AsyncBuffer {
public:
    BufferPromise(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type,
                const TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
                std::optional<ShardSpecBuffer> shard_parameters = std::nullopt, std::optional<bool> bottom_up = std::nullopt)
    {
        buffer_ = std::shared_ptr<tt::tt_metal::Buffer>(
            new tt::tt_metal::Buffer(device, size, page_size, buffer_type, buffer_layout, shard_parameters, bottom_up, false),
            [device, buffer_type](tt::tt_metal::Buffer* raw_buffer) {
                TT_FATAL(device->allocator_ != nullptr, "Expected allocator to be initialized!");

                device->push_work([device, buffer_type, raw_buffer] () mutable {
                    delete raw_buffer;
                });
            }
        );

        this->allocate();
    }

    operator std::shared_ptr<tt::tt_metal::Buffer>() {
        return buffer_;
    }

private:
    void allocate() {
        auto* device = buffer_->device();
        TT_ASSERT(device);
        device->push_work([buffer = buffer_] () mutable {
            buffer->allocate();
        });
    }

    std::shared_ptr<tt::tt_metal::Buffer> buffer_;
};

static std::ostream &operator<<(std::ostream &os, const CoreGrid &core_grid) {
    os << "ttnn.CoreGrid(x=" <<core_grid.x<<", y="<<core_grid.y<<")";
    return os;
}

}  // namespace types

using namespace types;

}  // namespace ttnn
