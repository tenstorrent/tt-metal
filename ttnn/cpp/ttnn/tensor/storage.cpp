// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

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
    specs(std::move(specs_)), mesh_buffer(std::move(mesh_buffer_)) {}

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

MultiDeviceHostStorage::MultiDeviceHostStorage(DistributedHostBuffer distributed_buffer, TensorSpec spec) :
    storage_(std::move(distributed_buffer)) {
    specs_.push_back(std::move(spec));
}

MultiDeviceHostStorage::MultiDeviceHostStorage(std::vector<HostBuffer> buffers, std::vector<TensorSpec> specs) :
    storage_(std::move(buffers)), specs_(std::move(specs)) {}

HostBuffer MultiDeviceHostStorage::get_buffer(int buffer_index) const {
    return std::visit(
        tt::stl::overloaded{
            [buffer_index](const DistributedHostBuffer& distributed_buffer) {
                TT_FATAL(
                    distributed_buffer.shape().mesh_size() == 1 && buffer_index == 0,
                    "Only support getting shard 0 from distributed buffer of unit shape");
                return *distributed_buffer.get_shard(
                    distributed::MeshCoordinate::zero_coordinate(distributed_buffer.shape().dims()));
            },
            [buffer_index](const std::vector<HostBuffer>& host_buffers) {
                TT_FATAL(
                    buffer_index < host_buffers.size(),
                    "Buffer index {} out of bounds {}",
                    buffer_index,
                    host_buffers.size());
                return host_buffers[buffer_index];
            },
        },
        storage_);
}

bool MultiDeviceHostStorage::is_distributed_buffer() const {
    return std::holds_alternative<DistributedHostBuffer>(storage_);
}

const DistributedHostBuffer& MultiDeviceHostStorage::get_distributed_buffer() const {
    return std::get<DistributedHostBuffer>(storage_);
}

const std::vector<HostBuffer>& MultiDeviceHostStorage::get_host_buffers() const {
    return std::get<std::vector<HostBuffer>>(storage_);
}

TensorSpec MultiDeviceHostStorage::get_tensor_spec(int spec_index) const {
    TT_FATAL(spec_index < specs_.size(), "Spec for device {} not found in spec list", spec_index);
    return specs_[spec_index];
}

size_t MultiDeviceHostStorage::num_buffers() const {
    return std::visit(
        tt::stl::overloaded{
            [](const DistributedHostBuffer& distributed_buffer) { return distributed_buffer.shape().mesh_size(); },
            [](const std::vector<HostBuffer>& host_buffers) { return host_buffers.size(); },
        },
        storage_);
}
bool MultiDeviceHostStorage::is_allocated() const {
    return std::visit(
        tt::stl::overloaded{
            [](const DistributedHostBuffer& distributed_buffer) { return distributed_buffer.is_allocated(); },
            [](const std::vector<HostBuffer>& host_buffers) {
                return std::all_of(
                    host_buffers.begin(), host_buffers.end(), [](const auto& buffer) { return buffer.is_allocated(); });
            },
        },
        storage_);
}
void MultiDeviceHostStorage::deallocate() {
    std::visit(
        tt::stl::overloaded{
            [](DistributedHostBuffer& distributed_buffer) { distributed_buffer.deallocate(); },
            [](std::vector<HostBuffer>& host_buffers) {
                for (auto& buffer : host_buffers) {
                    buffer.deallocate();
                }
            },
        },
        storage_);
}

}  // namespace tt::tt_metal
