// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt-metalium/mesh_coord.hpp"

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {

DeviceStorage::DeviceStorage(std::shared_ptr<Buffer> buffer_) { buffer = std::move(buffer_); }

MemoryConfig DeviceStorage::memory_config() const {
    auto* buffer_to_use = get_buffer();

    std::optional<ShardSpec> shard_spec = std::nullopt;

    if (is_sharded(buffer_to_use->buffer_layout())) {
        shard_spec = buffer_to_use->shard_spec().tensor_shard_spec;
    }
    return MemoryConfig{
        buffer_to_use->buffer_layout(),
        buffer_to_use->buffer_type(),
        shard_spec,
    };
}

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
    std::vector<std::pair<distributed::MeshCoordinate, TensorSpec>> specs_) :
    specs(std::move(specs_)), mesh_buffer(std::move(mesh_buffer_)) {
    TT_FATAL(
        std::all_of(
            specs.begin(), specs.end(), [this](const auto& spec) { return spec.second == specs.front().second; }),
        "All specs in device storage must be the same");
}

Buffer* DeviceStorage::get_buffer() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->get_reference_buffer();
    }
    TT_FATAL(this->buffer != nullptr, "Buffer is not allocated");
    return this->buffer.get();
}

std::shared_ptr<distributed::MeshBuffer> DeviceStorage::get_mesh_buffer() const {
    TT_FATAL(mesh_buffer != nullptr, "Mesh buffer is not allocated");
    return mesh_buffer;
}

bool DeviceStorage::is_allocated() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->is_allocated();
    }
    return this->buffer != nullptr && this->buffer->is_allocated();
}

IDevice* DeviceStorage::get_device() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->device();
    }
    TT_FATAL(this->buffer != nullptr, "Buffer is not allocated");
    return this->buffer->device();
}

void DeviceStorage::update_specs(const TensorSpec& new_spec) {
    for (auto& [_, spec] : this->specs) {
        spec = new_spec;
    }
}

bool DeviceStorage::is_uniform_storage() const {
    if (mesh_buffer.get() == nullptr) {
        return true;
    }
    return specs.size() == mesh_buffer->device()->num_devices() &&
           std::all_of(specs.begin(), specs.end(), [this](const auto& spec) { return spec.second == specs[0].second; });
}

MultiDeviceHostStorage::MultiDeviceHostStorage(std::vector<HostBuffer> buffers, std::vector<TensorSpec> specs) :
    buffers_(std::move(buffers)), specs_(std::move(specs)) {
    TT_FATAL(
        std::all_of(specs_.begin(), specs_.end(), [this](const auto& spec) { return spec == specs_.front(); }),
        "All specs in multi-device host storage must be the same");
}

HostBuffer MultiDeviceHostStorage::get_buffer(int buffer_index) const {
    TT_FATAL(buffer_index < buffers_.size(), "Buffer not found for buffer_index {}", buffer_index);
    return buffers_[buffer_index];
}

TensorSpec MultiDeviceHostStorage::get_tensor_spec(int spec_index) const {
    TT_FATAL(spec_index < specs_.size(), "Spec for device {} not found in spec list", spec_index);
    return specs_[spec_index];
}

size_t MultiDeviceHostStorage::num_buffers() const { return buffers_.size(); }

}  // namespace tt::tt_metal
