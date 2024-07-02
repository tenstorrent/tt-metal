// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/detail/tt_metal.hpp"
#include "ttnn/experimental/tensor/tensor.hpp"
#include "ttnn/experimental/tensor/types.hpp"

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

struct CoreGrid {
    std::size_t x;
    std::size_t y;

    CoreGrid(std::size_t x, std::size_t y) : x(x), y(y) {}
    CoreCoord to_CoreCoord(){
        return CoreCoord(int(x), int(y));
    }
};

// This buffer class is compatible with multithreaded runtime (which lives in tt_eager)
// It is derived from the tt_metal::Buffer class, but defines its own asynchronous allocation functions
class Buffer : public tt::tt_metal::Buffer {
    public:
        Buffer(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type,
                const TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
                std::optional< ShardSpecBuffer> shard_parameters = std::nullopt
            ) : tt::tt_metal::Buffer(device, size, page_size, buffer_type, buffer_layout, shard_parameters, false) {
                this->allocate();
            }
        ~Buffer() {
            this->deallocate();
        }
    private:
        void allocate() {
            TT_ASSERT(this->device());
            this->device()->push_work([this] () mutable {
                bool bottom_up = this->buffer_type() == BufferType::DRAM;
                tt::tt_metal::detail::AllocateBuffer(this, bottom_up);

            });
        }
        void deallocate() {
            if (this->device() == nullptr or not this->device()->initialized_ or this->size() == 0) {
                return;
            }
            this->set_size(0);
            TT_ASSERT(this->device()->allocator_ != nullptr, "Expected allocator to be initialized!");
            this->device()->push_work([this] () mutable {
                tt::tt_metal::detail::DeallocateBuffer(this);
            });
        }
};

static std::ostream &operator<<(std::ostream &os, const CoreGrid &core_grid) {
    os << "ttnn.CoreGrid(x=" <<core_grid.x<<", y="<<core_grid.y<<")";
    return os;
}

}  // namespace types

using namespace types;

}  // namespace ttnn
