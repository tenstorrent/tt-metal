// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/storage.hpp"
#include "tt-metalium/mesh_coord.hpp"

namespace tt::tt_metal {

TensorSpecMapping::TensorSpecMapping(
    const distributed::MeshCoordinateRange& mesh_range,
    const std::vector<std::pair<TensorSpec, distributed::MeshCoordinate>>& specs) {
    const bool uniform_specs = std::adjacent_find(specs.begin(), specs.end(), [](const auto& a, const auto& b) {
                                   return a.first != b.first;
                               }) == specs.end();
    const bool uniform_range = [&mesh_range, &specs]() {
        if (mesh_range.size() != specs.size()) {
            return false;
        }
        auto specs_it = specs.begin();
        for (const auto& coord : mesh_range) {
            if (coord != specs_it->second) {
                return false;
            }
            ++specs_it;
        }
        return true;
    }();

    // Same tensor spec covers the entire `mesh_range` uniformly.
    if (uniform_specs && uniform_range) {
        num_shards_ = specs.size();
        spec_mapping_ = {std::make_pair(specs.front().first, distributed::MeshCoordinateRangeSet(mesh_range))};
    } else {
        // Slow path for heterogeneous shards.
        num_shards_ = specs.size();

        auto add_to_specs = [this](const TensorSpec& spec, const distributed::MeshCoordinate& coord) {
            // If `spec` was already seen, add the coordinate to the existing range set.
            // If not, create a new range set with the coordinate.
            auto existing_it = std::find_if(
                spec_mapping_.begin(), spec_mapping_.end(), [&spec](const auto& p) { return p.first == spec; });
            if (existing_it != spec_mapping_.end()) {
                existing_it->second.merge(distributed::MeshCoordinateRange(coord, coord));
            } else {
                spec_mapping_.push_back(std::make_pair(
                    spec, distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(coord, coord))));
            }
        };
        for (const auto& [spec, coord] : specs) {
            add_to_specs(spec, coord);
        }
    }
}

TensorSpecMapping::TensorSpecMapping(const TensorSpec& spec, const distributed::MeshCoordinateRange& range) :
    spec_mapping_({std::make_pair(spec, distributed::MeshCoordinateRangeSet(range))}), num_shards_(range.size()) {}

TensorSpecMapping::TensorSpecMapping(const TensorSpec& spec, const distributed::MeshCoordinate& coord) :
    TensorSpecMapping(spec, distributed::MeshCoordinateRange(coord, coord)) {}

size_t TensorSpecMapping::num_shards() const { return num_shards_; }

bool TensorSpecMapping::is_uniform_spec() const { return spec_mapping_.size() == 1; }

std::vector<std::pair<TensorSpec, distributed::MeshCoordinate>> TensorSpecMapping::flatten() const {
    std::vector<std::pair<TensorSpec, distributed::MeshCoordinate>> result;
    result.reserve(num_shards());

    for (const auto& [spec, range_set] : spec_mapping_) {
        for (const auto& coord_range : range_set.ranges()) {
            for (const auto& coord : coord_range) {
                result.emplace_back(spec, coord);
            }
        }
    }

    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    return result;
}

void TensorSpecMapping::update_uniform_spec(const TensorSpec& new_spec) {
    TT_FATAL(is_uniform_spec(), "Cannot update non-uniform spec mapping");
    spec_mapping_.front().first = new_spec;
}

DeviceStorage::DeviceStorage(std::shared_ptr<Buffer> buffer_) { buffer = std::move(buffer_); }

MemoryConfig DeviceStorage::memory_config() const {
    auto* buffer_to_use = get_buffer();

    std::optional<ShardSpec> shard_spec = std::nullopt;

    if (is_sharded(buffer_to_use->buffer_layout())) {
        shard_spec = buffer_to_use->shard_spec().tensor_shard_spec;
    }
    return MemoryConfig{
        .memory_layout = buffer_to_use->buffer_layout(),
        .buffer_type = buffer_to_use->buffer_type(),
        .shard_spec = shard_spec,
    };
}

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
    DistributedTensorConfig strategy_,
    TensorSpecMapping spec_mapping_) :
    strategy(std::move(strategy_)), spec_mapping(std::move(spec_mapping_)), mesh_buffer(std::move(mesh_buffer_)) {}

void DeviceStorage::insert_buffer(const std::shared_ptr<Buffer>& buffer_) { this->buffer = buffer_; }

Buffer* DeviceStorage::get_buffer() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->get_reference_buffer();
    }
    TT_FATAL(this->buffer != nullptr, "Buffer is not allocated");
    return this->buffer.get();
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

}  // namespace tt::tt_metal
