#include "binary_constraints.hpp"

std::vector<OpConstraint> EltwiseOpConstraintsBuilder::build_constraints() {
    if (can_build_constraints() == false) {
        throw std::runtime_error("Cannot build constraints, missing required parameters");
    }

    std::vector<Layout> tile_layouts_a = {Layout::ROW_MAJOR, Layout::TILE};
    if (tile_layout_a.has_value()) {
        tile_layouts_a = {tile_layout_a.value()};
    }
    std::vector<Layout> tile_layouts_b = {Layout::ROW_MAJOR, Layout::TILE};
    if (tile_layout_b.has_value()) {
        tile_layouts_b = {tile_layout_b.value()};
    }
    std::vector<Layout> tile_layouts_o = {Layout::ROW_MAJOR, Layout::TILE};
    if (tile_layout_o.has_value()) {
        tile_layouts_o = {tile_layout_b.value()};
    }
    // Only two for now. TODO: add other storage types.
    std::vector<StorageType> storage_types_a = {StorageType::OWNED, StorageType::DEVICE};
    if (storage_type_a.has_value()) {
        storage_types_a = {storage_type_a.value()};
    }
    std::vector<StorageType> storage_types_b = {StorageType::OWNED, StorageType::DEVICE};
    if (storage_type_b.has_value()) {
        storage_types_b = {storage_type_b.value()};
    }
    std::vector<StorageType> storage_types_o = {StorageType::OWNED, StorageType::DEVICE};
    if (storage_type_o.has_value()) {
        storage_types_o = {storage_type_a.value()};
    }

    std::vector<OpConstraint> constraints;
    // Currently we are only looking at softmax for one input.
    for (const auto& tile_layout_a : tile_layouts_a) {
        for (const auto& storage_type_a : storage_types_a) {
            for (const auto& tile_layout_b : tile_layouts_b) {
                for (const auto& storage_type_b : storage_types_b) {
                    for (const auto& tile_layout_o : tile_layouts_o) {
                        for (const auto& storage_type_o : storage_types_o) {
                            const auto constraint = OpConstraint(
                                data_type_a.value(),
                                tile_layout_a,
                                storage_type_a,
                                data_type_b.value(),
                                tile_layout_b,
                                storage_type_b,
                                data_type_o.value(),
                                tile_layout_o,
                                storage_type_o);
                            if (is_valid_external_constraint(constraint) && is_valid_op_constraint(constraint)) {
                                constraints.emplace_back(constraint);
                            }
                        }
                    }
                }
            }
        }
    }

    return std::move(constraints);
}

bool EltwiseOpConstraintsBuilder::is_valid_op_constraint(const OpConstraint& constraint) const {
    const tt::tt_metal::Layout c_tile_layout_a = constraint.getTileLayoutA().value();
    const tt::tt_metal::Layout c_tile_layout_b = constraint.getTileLayoutB().value();
    const tt::tt_metal::DataType data_type_a = constraint.getDataTypeA().value();
    const tt::tt_metal::DataType data_type_b = constraint.getDataTypeB().value();
    if (memory_config_a.is_sharded() &&
        !is_sharded_tensor_valid(memory_config_a, shape_a, c_tile_layout_a, data_type_a)) {
        return false;
    }
    if (memory_config_b.is_sharded() &&
        !is_sharded_tensor_valid(memory_config_b, shape_b, c_tile_layout_b, data_type_b)) {
        return false;
    }
    if (c_tile_layout_a != tt::tt_metal::Layout::TILE) {
        return false;
    }
    if (c_tile_layout_b != tt::tt_metal::Layout::TILE) {
        return false;
    }

    return true;
}

bool ElementWiseMultiCoreConstraintsBuilder::check_input_parameters(
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::Shape& input_shape_b,
    const tt::tt_metal::MemoryConfig& memory_config_b) {
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    if (height_a == height_b and width_a == width_b) {
        return true;
    }
    return false;
}

bool BroadcastWidthMultiCoreConstraintsBuilder::check_input_parameters(
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::Shape& input_shape_b,
    const tt::tt_metal::MemoryConfig& memory_config_b) {
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    if (width_b == 1 && height_b > 1) {
        return true;
    }
    return false;
}

std::unique_ptr<EltwiseOpConstraintsBuilder> EltwiseOpConstraintsFactory::Make(
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::Shape& input_shape_b,
    const tt::tt_metal::MemoryConfig& memory_config_b,
    const tt::tt_metal::MemoryConfig& memory_config_o,
    const CoreCoord& chip_grid) {
    if (!OpConstraintsFactory::can_fit_op_on_chip(memory_config_a, chip_grid)) {
        return nullptr;
    }
    if (!OpConstraintsFactory::can_fit_op_on_chip(memory_config_b, chip_grid)) {
        return nullptr;
    }
    if (!OpConstraintsFactory::can_fit_op_on_chip(memory_config_o, chip_grid)) {
        return nullptr;
    }
    auto eltwise_op_type =
        GetEltwiseOpType(input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
    switch (eltwise_op_type) {
        case EltwiseOpTypes::ElementWiseMultiCore:
            return std::make_unique<ElementWiseMultiCoreConstraintsBuilder>(
                input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
        case EltwiseOpTypes::BroadcastWidthMultiCore:
            return std::make_unique<BroadcastWidthMultiCoreConstraintsBuilder>(
                input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
        case EltwiseOpTypes::BroadcastHeightMultiCore:                  // not implemented yet
        case EltwiseOpTypes::BroadcastHeightAndWidthMultiCore:          // not implemented yet
        case EltwiseOpTypes::BroadcastHeightMultiCoreSharded:           // not implemented yet
        case EltwiseOpTypes::BroadcastHeightMultiCoreShardedOptimized:  // not implemented yet
        default: return nullptr;
    }
};

EltwiseOpTypes EltwiseOpConstraintsFactory::GetEltwiseOpType(
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::Shape& input_shape_b,
    const tt::tt_metal::MemoryConfig& memory_config_b,
    const tt::tt_metal::MemoryConfig& memory_config_o) {
    // void BinaryDeviceOperation::validate_on_program_cache_hit(
    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];
    // Input shape b must be the same as or broadcastable to input shape a
    if (batch_size_0_a != batch_size_0_b) {
        if (!(batch_size_0_a > batch_size_0_b and batch_size_0_b == 1)) {
            // "ttnn::operations::binary::BinaryDeviceOperation: batch size mismatch");
            return EltwiseOpTypes::NotSupported;
        }
    }
    if (batch_size_1_a != batch_size_1_b) {
        if (!(batch_size_1_a > batch_size_1_b and batch_size_1_b == 1)) {
            // "ttnn::operations::binary::BinaryDeviceOperation: batch size mismatch");
            return EltwiseOpTypes::NotSupported;
        }
    }
    if (height_a != height_b) {
        if (!(height_a > height_b and height_b == 1)) {
            // "ttnn::operations::binary::BinaryDeviceOperation: height mismatch");
            return EltwiseOpTypes::NotSupported;
        }
    }
    if (width_a != width_b) {
        if (!(width_a > width_b and width_b == 1)) {
            // "ttnn::operations::binary::BinaryDeviceOperation: width mismatch");
            return EltwiseOpTypes::NotSupported;
        }
    }
    if (memory_config_a.is_sharded()) {
        if (memory_config_a.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            // If we aren't height sharded, we require all sharding schemes to match until we add blocked
            // reader/writers for width and block sharding
            if (!(memory_config_b.is_sharded())) {
                return EltwiseOpTypes::NotSupported;
            }
            if (memory_config_a.shard_spec.value().grid.ranges().size() != 1) {
                return EltwiseOpTypes::NotSupported;
            }
        }
        if (memory_config_b.is_sharded()) {
            if (memory_config_a.memory_layout != memory_config_b.memory_layout) {
                return EltwiseOpTypes::NotSupported;
            }
            if (memory_config_a.shard_spec.value() != memory_config_b.shard_spec.value()) {
                return EltwiseOpTypes::NotSupported;
            }
        }
        if (memory_config_o.is_sharded()) {
            if (memory_config_a.memory_layout != memory_config_o.memory_layout) {
                return EltwiseOpTypes::NotSupported;
            }
        } else {
            if (memory_config_o.memory_layout != TensorMemoryLayout::INTERLEAVED) {
                return EltwiseOpTypes::NotSupported;
            }
        }
    } else if (memory_config_b.is_sharded()) {
        if (memory_config_b.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            return EltwiseOpTypes::NotSupported;
        }
        if (memory_config_a.memory_layout != TensorMemoryLayout::INTERLEAVED) {
            return EltwiseOpTypes::NotSupported;
        }
        if (memory_config_o.is_sharded()) {
            if (memory_config_b.memory_layout != memory_config_o.memory_layout) {
                return EltwiseOpTypes::NotSupported;
            }
        } else {
            if (memory_config_b.memory_layout != TensorMemoryLayout::INTERLEAVED) {
                return EltwiseOpTypes::NotSupported;
            }
        }
    } else {
        if (memory_config_a.memory_layout != TensorMemoryLayout::INTERLEAVED) {
            return EltwiseOpTypes::NotSupported;
        }
        if (memory_config_b.memory_layout != TensorMemoryLayout::INTERLEAVED) {
            return EltwiseOpTypes::NotSupported;
        }
        if (memory_config_o.is_sharded()) {
            if (memory_config_o.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                return EltwiseOpTypes::NotSupported;
            }
            // TODO: Check if we need this. This will have the consequence that building the constraint framework will
            // require a TT card in the system.
            /*uint32_t num_blocks = Volume(input_shape_a) / input_shape_a[-1] / tt::constants::TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            if (num_blocks < num_cores or num_blocks % num_cores != 0)
            {
                return EltwiseOpTypes::NotSupported;
            }*/
        } else {
            if (memory_config_o.memory_layout != TensorMemoryLayout::INTERLEAVED) {
                return EltwiseOpTypes::NotSupported;
            }
        }
    }
    if (ElementWiseMultiCoreConstraintsBuilder::check_input_parameters(
            input_shape_a, memory_config_a, input_shape_b, memory_config_b)) {
        return EltwiseOpTypes::ElementWiseMultiCore;
    } else if (BroadcastWidthMultiCoreConstraintsBuilder::check_input_parameters(
                   input_shape_a, memory_config_a, input_shape_b, memory_config_b)) {
        return EltwiseOpTypes::BroadcastWidthMultiCore;
    }
    std::cout << "EltwiseOpTypes::NotSupported" << std::endl;
    // todo other op flavors

    return EltwiseOpTypes::NotSupported;
}
