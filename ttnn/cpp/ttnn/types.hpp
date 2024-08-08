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
using DeviceGrid = tt::tt_metal::DeviceGrid;
using DeviceIds = tt::tt_metal::DeviceIds;
using DeviceMesh = tt::tt_metal::DeviceMesh;
using DeviceMeshView = tt::tt_metal::DeviceMeshView;

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
class Buffer : public tt::tt_metal::Buffer, public std::enable_shared_from_this<Buffer> {
    public:
        static std::shared_ptr<Buffer> Create(Device* device, uint64_t size, uint64_t page_size, const BufferType buffer_type,
                const TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
                std::optional<ShardSpecBuffer> shard_parameters = std::nullopt
            ) {
                // Factory Method for creating Buffer objects. Will return std::shared_ptr<Buffer>.
                // This is the only interface for creating Buffers, to ensure that users don't use
                // the Buffer class without shared_ptr, since this class uses enable_shared_from_this.
                std::shared_ptr<Buffer> buffer_instance(new Buffer(device, size, page_size, buffer_type, buffer_layout, shard_parameters));
                buffer_instance->allocate(); // Asynchronously allocate the buffer
                return buffer_instance;
            }
        ~Buffer() {
            this->deallocate(); // Asynchronously clean up device state used by this buffer
        }
        uint32_t address() {
            this->allocated_.wait(false); // Wait until buffer is allocated
            return static_cast<tt::tt_metal::Buffer*>(this)->address();
        }

    private:
        Buffer(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type,
                const TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
                std::optional< ShardSpecBuffer> shard_parameters = std::nullopt
            ) : tt::tt_metal::Buffer(device, size, page_size, buffer_type, buffer_layout, shard_parameters, false) {
                // Private constructor used by factory method. Each buffer is given a unique id.
                this->buffer_instance_id = Buffer::buf_id++;
            }

        void allocate() {
            TT_ASSERT(this->device());
            this->device()->push_work([shared_buffer_ptr = shared_from_this()] () mutable {
                bool bottom_up = shared_buffer_ptr->buffer_type() == BufferType::DRAM;
                tt::tt_metal::detail::AllocateBuffer(shared_buffer_ptr.get(), bottom_up);
                // Notify address getter that this buffer was allocated
                shared_buffer_ptr->allocated_ = true;
                shared_buffer_ptr->allocated_.notify_one();
                // The address inserted here, will be used during asynchronous deallocate
                Buffer::buf_id_to_address_map.insert({shared_buffer_ptr->buffer_instance_id, shared_buffer_ptr->address()});
            });
        }

        void deallocate() {
            if (this->device() == nullptr or not this->device()->initialized_ or this->size() == 0) {
                return;
            }
            this->set_size(0);
            TT_ASSERT(this->device()->allocator_ != nullptr, "Expected allocator to be initialized!");
            // Extract the required buffer attributes from main thread (these are guaranteed to be correctly populated) and send to worker
            this->device()->push_work([dev = this->device(), id = this->buffer_instance_id, type = this->buffer_type()] () mutable {
                // At this point, the address for this buffer has made it to buf_id_to_address_map, since the worker has allocated the buffer.
                tt::tt_metal::allocator::deallocate_buffer(*(dev->allocator_), Buffer::buf_id_to_address_map.at(id), type);
                Buffer::buf_id_to_address_map.erase(id);
            });
        }
    private:
        // Thread local variable used to extract buffer address from its id (used for asynchronous deallocation)
        inline static thread_local std::unordered_map<uint64_t, uint32_t> buf_id_to_address_map = {};
        // Global buffer id counter. Used to assign each buffer its own id
        inline static uint64_t buf_id = 0;
        // The id for this buffer instance
        uint64_t buffer_instance_id = 0;
        std::atomic<bool> allocated_ = false;
};

static std::ostream &operator<<(std::ostream &os, const CoreGrid &core_grid) {
    os << "ttnn.CoreGrid(x=" <<core_grid.x<<", y="<<core_grid.y<<")";
    return os;
}

}  // namespace types

using namespace types;

}  // namespace ttnn
