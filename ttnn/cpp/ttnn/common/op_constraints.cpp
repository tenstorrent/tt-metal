#include "ttnn/common/op_constraints.hpp"

#include <optional>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor_impl.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor_utils.hpp"

bool OpConstraintsBuilder::is_valid_external_constraint(const OpConstraint& constraint) const {
    if (data_type_a.has_value() && constraint.getDataTypeA().value() != data_type_a.value()) {
        return false;
    }
    if (tile_layout_a.has_value() && constraint.getTileLayoutA().value() != tile_layout_a.value()) {
        return false;
    }
    if (storage_type_a.has_value() && constraint.getStorageTypeA().value() != storage_type_a.value()) {
        return false;
    }
    if (data_type_b.has_value() && constraint.getDataTypeB().value() != data_type_b.value()) {
        return false;
    }
    if (tile_layout_b.has_value() && constraint.getTileLayoutB().value() != tile_layout_b.value()) {
        return false;
    }
    if (storage_type_b.has_value() && constraint.getStorageTypeB().value() != storage_type_b.value()) {
        return false;
    }
    if (data_type_o.has_value() && constraint.getDataTypeO().value() != data_type_o.value()) {
        return false;
    }
    if (tile_layout_o.has_value() && constraint.getTileLayoutO().value() != tile_layout_o.value()) {
        return false;
    }
    if (storage_type_o.has_value() && constraint.getStorageTypeO().value() != storage_type_o.value()) {
        return false;
    }
    return true;
}

bool OpConstraintsBuilder::is_sharded_tensor_valid(
    const MemoryConfig& memory_config,
    const ttnn::Shape& shape,
    const Layout& layout,
    const DataType& data_type) const {
    uint32_t total_height = tt::tt_metal::compute_volume(shape) / shape[-1];
    uint32_t total_width = shape[-1];
    if (memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        if (total_width != shard_shape[1]) {
            return false;
        }
        uint32_t num_shards = tt::div_up(total_height, shard_shape[0]);
        uint32_t num_cores = memory_config.shard_spec.value().grid.num_cores();
        if (num_shards > num_cores) {
            return false;
        }
    } else if (memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        if (total_height != shard_shape[0]) {
            return false;
        }
        uint32_t num_shards = tt::div_up(total_width, shard_shape[1]);
        uint32_t num_cores = shard_spec.grid.num_cores();
        if (num_shards > num_cores) {
            return false;
        }
    } else if (memory_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        if (shard_spec.grid.ranges().size() != 1) {
            return false;
        }
        uint32_t num_shards_along_height = tt::div_up(total_height, shard_shape[0]);
        uint32_t num_shards_along_width = tt::div_up(total_width, shard_shape[1]);

        // Additionally check that number of cores along height and width matches shard grid
        const CoreCoord shard_grid = shard_spec.grid.bounding_box().grid_size();
        if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
            if (num_shards_along_height > shard_grid.y) {
                return false;
            }
            if (num_shards_along_width > shard_grid.x) {
                return false;
            }
        } else {
            if (num_shards_along_height > shard_grid.x) {
                return false;
            }
            if (num_shards_along_width > shard_grid.y) {
                return false;
            }
        }
    } else {
        return false;
    }
    if (layout == Layout::TILE) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        if (!(shard_shape[0] % tt::constants::TILE_HEIGHT == 0 && shard_shape[1] % tt::constants::TILE_WIDTH == 0)) {
            return false;
        }
    } else if (layout == Layout::ROW_MAJOR) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        if (!(shard_shape[1] * tensor_impl::element_size_bytes(data_type) % sizeof(uint32_t) == 0)) {
            return false;
        }
    }
    if (!can_allocate_sharded_buffer(memory_config, shape, layout, data_type)) {
        return false;
    }
    return true;
}

bool OpConstraintsBuilder::can_build_constraints() const {
    return data_type_a.has_value() && data_type_b.has_value() && data_type_o.has_value();
}

// Setters for parameter a
OpConstraintsBuilder& OpConstraintsBuilder::setDataTypeA(tt::tt_metal::DataType dataType) {
    data_type_a = dataType;
    return *this;
}

OpConstraintsBuilder& OpConstraintsBuilder::setTileLayoutA(tt::tt_metal::Layout tileLayout) {
    tile_layout_a = tileLayout;
    return *this;
}

OpConstraintsBuilder& OpConstraintsBuilder::setStorageTypeA(tt::tt_metal::StorageType storageType) {
    storage_type_a = storageType;
    return *this;
}

// Setters for parameter b
OpConstraintsBuilder& OpConstraintsBuilder::setDataTypeB(tt::tt_metal::DataType dataType) {
    data_type_b = dataType;
    return *this;
}

OpConstraintsBuilder& OpConstraintsBuilder::setTileLayoutB(tt::tt_metal::Layout tileLayout) {
    tile_layout_b = tileLayout;
    return *this;
}

OpConstraintsBuilder& OpConstraintsBuilder::setStorageTypeB(tt::tt_metal::StorageType storageType) {
    storage_type_b = storageType;
    return *this;
}

// Setters for parameter output
OpConstraintsBuilder& OpConstraintsBuilder::setDataTypeO(tt::tt_metal::DataType dataType) {
    data_type_o = dataType;
    return *this;
}

OpConstraintsBuilder& OpConstraintsBuilder::setTileLayoutO(tt::tt_metal::Layout tileLayout) {
    tile_layout_o = tileLayout;
    return *this;
}

OpConstraintsBuilder& OpConstraintsBuilder::setStorageTypeO(tt::tt_metal::StorageType storageType) {
    storage_type_o = storageType;
    return *this;
}

uint32_t OpConstraintsBuilder::get_packed_size_in_bytes(
    const tt::tt_metal::DataType& data_type, const ttnn::Shape& shape) const {
    return tensor_impl::packed_buffer_size_bytes_wrapper(data_type, compute_buffer_size(shape, data_type));
}

bool OpConstraintsBuilder::can_allocate_sharded_buffer(
    const MemoryConfig& memory_config,
    const ttnn::Shape& shape,
    const Layout& layout,
    const DataType& data_type) const {
    TT_ASSERT(memory_config.is_sharded(), "This check should only be done for sharded buffers.");
    uint32_t buffer_size_bytes = get_packed_size_in_bytes(data_type, shape);
    auto page_shape = tensor_impl::get_sharded_page_shape(layout, data_type, memory_config.shard_spec.value().shape);

    auto width = shape[-1];
    auto other_dims = 1;
    for (int i = 0; i < shape.rank() - 1; i++) {
        other_dims *= shape[i];
    }

    std::array<uint32_t, 2> tensor2d_size = {other_dims / page_shape[0], width / page_shape[1]};
    std::optional<ShardSpecBuffer> shard_spec_buffer_opt =
        ShardSpecBuffer(memory_config.shard_spec.value(), page_shape, tensor2d_size);
    const auto& shard_params_page_shape = shard_spec_buffer_opt.value().page_shape;
    uint32_t page_size = tt::tt_metal::tensor_impl::get_page_size(data_type, layout, buffer_size_bytes, page_shape);
    if (buffer_size_bytes == 0 || page_size == 0) {
        return false;
    }
    if (buffer_size_bytes % page_size != 0) {
        return false;
    }
    if (page_size % sizeof(uint32_t) != 0) {
        return false;
    }

    if (is_sharded(memory_config.memory_layout)) {
        if (shard_spec_buffer_opt == std::nullopt) {
            return false;
        }
    } else if (memory_config.memory_layout == TensorMemoryLayout::SINGLE_BANK) {
        if (page_size != buffer_size_bytes) {
            return false;
        }
    }
    return true;
}
