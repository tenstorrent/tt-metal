// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/types.hpp"

namespace ttnn::types {

Buffer::Buffer(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type,
        const TensorMemoryLayout buffer_layout,
        std::optional< ShardSpecBuffer> shard_parameters
    ) : tt::tt_metal::Buffer(device, size, page_size, buffer_type, buffer_layout, shard_parameters, false) {
        this->buffer_id = get_unique_id(); // Each buffer has a unique ID
        this->allocate();
}

Buffer::~Buffer() {
    this->deallocate();
}

void Buffer::allocate() {
    TT_ASSERT(this->device());
    this->device()->push_work([this] () mutable {
        bool bottom_up = this->buffer_type() == BufferType::DRAM;
        tt::tt_metal::detail::AllocateBuffer(this, bottom_up);
        // The address inserted here, will be used during asynchronous deallocate
    });
}

void Buffer::deallocate() {
    if (this->device() == nullptr or not this->device()->initialized_ or this->size() == 0) {
        return;
    }
    this->set_size(0);
    TT_ASSERT(this->device()->allocator_ != nullptr, "Expected allocator to be initialized!");
    // Extract the required buffer attributes from main thread (these are guaranteed to be correctly populated) and send to worker
    this->device()->push_work([dev = this->device(), id = this->buffer_id, type = this->buffer_type(), buffer_address = this->address()] () mutable {
        tt::tt_metal::allocator::deallocate_buffer(*(dev->allocator_), buffer_address, type);
    });
}

uint32_t Buffer::get_unique_id() {
    return buf_id++;
}

std::atomic<uint32_t> Buffer::buf_id = 0;


}  // namespace ttnn
