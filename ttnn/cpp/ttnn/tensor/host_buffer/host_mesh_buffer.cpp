// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/span.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/assert.hpp>

#include "ttnn/tensor/host_buffer/host_mesh_buffer.hpp"

#include <functional>
#include <vector>

namespace tt::tt_metal {
namespace {

// Returns a vector of local buffers from the global buffers, validating that the local size and offset are consistent
// with the global size.
std::vector<HostBuffer> get_local_buffers(
    std::vector<HostBuffer>&& global_buffers, size_t local_size, size_t local_offset) {
    TT_FATAL(
        local_size + local_offset <= global_buffers.size(),
        "Local size {} and offset {} must be consistent with the global size {}",
        local_size,
        local_offset,
        global_buffers.size());
    std::vector<HostBuffer> local_buffers;
    local_buffers.reserve(local_size);
    for (size_t i = 0; i < local_size; ++i) {
        local_buffers.push_back(std::move(global_buffers.at(i + local_offset)));
    }
    return local_buffers;
}

}  // namespace

HostMeshBuffer HostMeshBuffer::create_replicated(HostBuffer host_buffer) {
    return HostMeshBuffer(Replicated{std::move(host_buffer)});
}

HostMeshBuffer HostMeshBuffer::create_sharded(std::vector<HostBuffer> host_buffers) {
    return HostMeshBuffer(Sharded{
        .global_size = host_buffers.size(),
        .local_size = host_buffers.size(),
        .local_offset = 0,
        .local_buffers = std::move(host_buffers)});
}

HostMeshBuffer HostMeshBuffer::create_sharded(
    std::vector<HostBuffer> global_buffers, size_t local_size, size_t local_offset) {
    const size_t global_size = global_buffers.size();
    return HostMeshBuffer(Sharded{
        .global_size = global_size,
        .local_size = local_size,
        .local_offset = local_offset,
        .local_buffers = get_local_buffers(std::move(global_buffers), local_size, local_offset)});
}

HostMeshBuffer HostMeshBuffer::create_sharded(std::vector<HostBuffer> buffers, const distributed::MeshShape& shape) {
    TT_FATAL(shape.mesh_size() == buffers.size(), "Shape {} and buffers size {} must be equal", shape, buffers.size());
    return HostMeshBuffer(Sharded{
        .global_size = shape.mesh_size(),
        .local_size = shape.mesh_size(),
        .local_offset = 0,
        .local_buffers = std::move(buffers)});
}

HostMeshBuffer HostMeshBuffer::create_sharded(
    std::vector<HostBuffer> buffers,
    const distributed::MeshShape& global_shape,
    const distributed::MeshShape& local_shape,
    const distributed::MeshCoordinate& local_offset) {
    TT_FATAL(
        global_shape.mesh_size() == buffers.size(),
        "Global shape {} and buffers size {} must be equal",
        global_shape,
        buffers.size());
    TT_FATAL(
        global_shape.dims() == local_shape.dims(),
        "Global shape {} and local shape {} must have the same dimensions",
        global_shape,
        local_shape);
    TT_FATAL(
        global_shape.dims() == local_offset.dims(),
        "Global shape {} and local offset {} must have the same dimensions",
        global_shape,
        local_offset);

    const size_t local_offset_linear_index = distributed::to_linear_index(global_shape, local_offset);
    return HostMeshBuffer(Sharded{
        .global_size = global_shape.mesh_size(),
        .local_size = local_shape.mesh_size(),
        .local_offset = local_offset_linear_index,
        .local_buffers = get_local_buffers(std::move(buffers), local_shape.mesh_size(), local_offset_linear_index)});
}

std::optional<HostBuffer> HostMeshBuffer::get_buffer(size_t linear_index) {
    return std::visit(
        tt::stl::overloaded{
            [](const Replicated& replicated) -> std::optional<HostBuffer> {
                return std::make_optional(replicated.buffer);
            },
            [linear_index](const Sharded& sharded) -> std::optional<HostBuffer> {
                TT_FATAL(
                    linear_index < sharded.global_size,
                    "Linear index {} is out of bounds for global size {}",
                    linear_index,
                    sharded.global_size);
                if (linear_index < sharded.local_offset || linear_index >= sharded.local_offset + sharded.local_size) {
                    return std::nullopt;
                } else {
                    const int local_index = linear_index - sharded.local_offset;
                    return std::make_optional(sharded.local_buffers.at(local_index));
                }
            }},
        data_);
}

void HostMeshBuffer::transform(std::function<HostBuffer(const HostBuffer& buffer, size_t linear_index)>& fn) {
    return std::visit(
        tt::stl::overloaded{
            [&fn](Replicated& replicated) { replicated.buffer = fn(replicated.buffer, 0); },
            [&fn](Sharded& sharded) {
                for (size_t i = 0; i < sharded.local_size; ++i) {
                    sharded.local_buffers.at(i) = fn(sharded.local_buffers.at(i), i);
                }
            }},
        data_);
}

void HostMeshBuffer::apply(std::function<void(const HostBuffer& buffer, size_t linear_index)>& fn) {
    return std::visit(
        tt::stl::overloaded{
            [&fn](Replicated& replicated) { fn(replicated.buffer, 0); },
            [&fn](Sharded& sharded) {
                for (size_t i = 0; i < sharded.local_size; ++i) {
                    fn(sharded.local_buffers.at(i), i);
                }
            }},
        data_);
}

bool HostMeshBuffer::is_allocated() const {
    return std::visit(
        tt::stl::overloaded{
            [](const Replicated& replicated) { return replicated.buffer.is_allocated(); },
            [](const Sharded& sharded) {
                return std::all_of(sharded.local_buffers.begin(), sharded.local_buffers.end(), [](const HostBuffer& b) {
                    return b.is_allocated();
                });
            }},
        data_);
}

void HostMeshBuffer::deallocate() {
    std::visit(
        tt::stl::overloaded{
            [](Replicated& replicated) { replicated.buffer.deallocate(); },
            [](Sharded& sharded) {
                for (auto& buffer : sharded.local_buffers) {
                    buffer.deallocate();
                }
            }},
        data_);
}

}  // namespace tt::tt_metal
