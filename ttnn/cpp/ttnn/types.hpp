// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {
namespace types {

using Device = tt::tt_metal::Device;
using MeshShape = tt::tt_metal::MeshShape;
using DeviceIds = tt::tt_metal::DeviceIds;
using MeshDevice = tt::tt_metal::MeshDevice;
using MeshDeviceView = tt::tt_metal::MeshDeviceView;

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

// Keep track of live buffers and the device addresses they were assigned.
// When a buffer is created, it is provided a buffer_id using get_buf_id().
// The address for this buffer is assigned to buffer_id when the buffer is asynchronously allocated.
// When the buffer destructor is called, or the buffer is asynchronously deallocated, the worker thread
// will look up the address for buffer_id to free memory on device.
class buffer_address_map {
    public:
        void insert(uint32_t buf_id, uint32_t buf_addr) {
            std::scoped_lock<std::mutex> lock(this->map_mutex);
            this->buf_id_to_address_map.insert({buf_id, buf_addr});
        }
        void erase(uint32_t buf_id) {
            std::scoped_lock<std::mutex> lock(this->map_mutex);
            this->buf_id_to_address_map.erase(buf_id);
        }
        uint32_t buffer_address(uint32_t buf_id) {
            std::scoped_lock<std::mutex> lock(this->map_mutex);
            return this->buf_id_to_address_map.at(buf_id);
        }
        uint32_t get_buf_id() {
            return buf_id++;
        }

    private:
    std::atomic<uint32_t> buf_id = 0;
    std::mutex map_mutex;
    std::unordered_map<uint32_t, uint32_t> buf_id_to_address_map = {};
};

inline buffer_address_map GLOBAL_BUFFER_ADDRESS_MAP;

// This buffer class is compatible with multithreaded runtime (which lives in tt_eager)
// It is derived from the tt_metal::Buffer class, but defines its own asynchronous allocation functions
class Buffer : public tt::tt_metal::Buffer {
    public:
        Buffer(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type,
                const TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
                std::optional<ShardSpecBuffer> shard_parameters = std::nullopt, std::optional<bool> bottom_up = std::nullopt
            ) : tt::tt_metal::Buffer(device, size, page_size, buffer_type, buffer_layout, shard_parameters, bottom_up, false) {
                this->buffer_id = GLOBAL_BUFFER_ADDRESS_MAP.get_buf_id(); // Each buffer has a unique ID
                this->allocate();
            }
        ~Buffer() {
            this->deallocate();
        }
    private:
        uint32_t buffer_id = 0;
        void allocate() {
            TT_ASSERT(this->device());
            DevicePushWork(this->device(), [this] () mutable {
                bool bottom_up = this->bottom_up_.value_or(this->is_dram());
                tt::tt_metal::detail::AllocateBuffer(this, bottom_up);
                // The address inserted here, will be used during asynchronous deallocate
                GLOBAL_BUFFER_ADDRESS_MAP.insert(this->buffer_id, this->address());

            });
        }
        void deallocate() {
            if (this->device() == nullptr or not DeviceIsInitialized(this->device()) or this->size() == 0) {
                return;
            }
            this->set_size(0);
            TT_ASSERT(DeviceGetAllocator(this->device()) != nullptr, "Expected allocator to be initialized!");
            // Extract the required buffer attributes from main thread (these are guaranteed to be correctly populated) and send to worker
            DevicePushWork(this->device(), [dev = this->device(), id = this->buffer_id, type = this->buffer_type()] () mutable {
                // At this point, the address for this buffer has made it to GLOBAL_BUFFER_ADDRESS_MAP, since the worker has allocated the buffer.
                tt::tt_metal::allocator::deallocate_buffer(*DeviceGetAllocator(dev), GLOBAL_BUFFER_ADDRESS_MAP.buffer_address(id), type);
                GLOBAL_BUFFER_ADDRESS_MAP.erase(id);
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
