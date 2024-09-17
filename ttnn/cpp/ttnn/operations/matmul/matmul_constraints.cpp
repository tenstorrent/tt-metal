#include "matmul_constraints.hpp"

std::vector<OpConstraint> MatmulOpConstraintsBuilder::build_constraints() {
    if (can_build_constraints() == false) {
        throw std::runtime_error("Cannot build constraints, missing required parameters");
    }

    std::vector<Layout> tile_layouts_a = {Layout::ROW_MAJOR, Layout::TILE};
    if (tile_layout_a.has_value()) {
        tile_layouts_a = {tile_layout_a.value()};
    }
    std::vector<Layout> tile_layouts_b = {Layout::ROW_MAJOR, Layout::TILE};
    if (tile_layout_b.has_value()) {
        tile_layouts_a = {tile_layout_b.value()};
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

bool MatmulOpConstraintsBuilder::is_valid_op_constraint(const OpConstraint& constraint) const {
    const tt::tt_metal::Layout c_tile_layout_a = constraint.getTileLayoutA().value();
    const tt::tt_metal::Layout c_tile_layout_b = constraint.getTileLayoutB().value();
    const tt::tt_metal::DataType data_type_a = constraint.getDataTypeA().value();
    const tt::tt_metal::DataType data_type_b = constraint.getDataTypeB().value();
    if (c_tile_layout_a != Layout::TILE || c_tile_layout_b != Layout::TILE) {
        return false;
    }
    if (memory_config_a.is_sharded() &&
        !is_sharded_tensor_valid(memory_config_a, shape_a, c_tile_layout_a, data_type_a)) {
        return false;
    }
    if (memory_config_b.is_sharded() &&
        !is_sharded_tensor_valid(memory_config_b, shape_b, c_tile_layout_b, data_type_b)) {
        return false;
    }
    if (!is_floating_point(data_type_a)) {
        return false;
    }
    return true;
}

std::unique_ptr<MatmulOpConstraintsBuilder> MatmulOpConstraintsFactory::Make(
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::Shape& input_shape_b,
    tt::tt_metal::MemoryConfig& memory_config_b,
    tt::tt_metal::MemoryConfig& memory_config_o,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config,
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
    auto matmul_op_type = GetMatmulOpType(
        input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o, program_config);
    switch (matmul_op_type) {
        case MatmulOpTypes::MatmulMultiCoreReuseMultiCast:
            return std::make_unique<MatmulMultiCoreReuseMultiCastConstraintsBuilder>(
                input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
        case MatmulOpTypes::MatmulMultiCoreReuseMultiCast1D:
            return std::make_unique<MatmulMultiCoreReuseMultiCast1DConstraintsBuilder>(
                input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
        case MatmulOpTypes::MatmulMultiCore:                           // not implemented yet
        case MatmulOpTypes::MatmulMultiCoreNonOptimizedReuse:          // not implemented yet
        case MatmulOpTypes::MatmulMultiCoreReuse:                      // not implemented yet
        case MatmulOpTypes::MatmulMultiCoreReuseMultiCastDRAMSharded:  // not implemented yet
        default: return nullptr;
    }
};

const uint32_t MatmulOpConstraintsFactory::Volume(const ttnn::Shape& shape) {
    auto rank = shape.rank();
    auto volume = 1;
    for (auto index = 0; index < rank; index++) {
        volume *= shape.operator[](index);
    }
    return volume;
}

MatmulOpTypes MatmulOpConstraintsFactory::GetMatmulOpType(
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::Shape& input_shape_b,
    const tt::tt_metal::MemoryConfig& memory_config_b,
    const tt::tt_metal::MemoryConfig& memory_config_o,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config) {
    if (input_shape_a[-1] != input_shape_b[-2]) {
        return MatmulOpTypes::NotSupported;
    }
    if (input_shape_a.rank() != input_shape_b.rank()) {
        return MatmulOpTypes::NotSupported;
    }
    for (auto i = 0; i < input_shape_a.rank() - 2; i++) {
        if (input_shape_a[i] != input_shape_b[i]) {
            return MatmulOpTypes::NotSupported;
        }
    }
    MatmulOpTypes matmul_op_type = MatmulOpTypes::NotSupported;
    std::visit(
        [&](const auto& program_config) {
            using T = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<T, ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (program_config.mcast_in0) {
                    if (memory_config_a.is_sharded()) {
                        if (!program_config.fuse_batch) {
                            return;
                        }
                        if (memory_config_a.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                            return;
                        }
                        if (memory_config_o.is_sharded()) {
                            if (memory_config_a.memory_layout != memory_config_o.memory_layout) {
                                return;
                            }
                        }
                        uint32_t M = (program_config.fuse_batch ? Volume(input_shape_a) / input_shape_a[-1]
                                                                : input_shape_a[-2]) /
                                     tt::constants::TILE_HEIGHT;
                        uint32_t K = input_shape_a[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = memory_config_a.shard_spec.value().shape;

                        if (M != per_core_M) {
                            return;
                        }
                        if (per_core_M != (shard_shape[0] / tt::constants::TILE_HEIGHT)) {
                            return;
                        }
                        if (K % program_config.in0_block_w != 0) {
                            return;
                        }
                        if ((shard_shape[1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w != 0) {
                            return;
                        }
                    }
                    if (memory_config_o.is_sharded()) {
                        if (memory_config_o.memory_layout != TensorMemoryLayout::WIDTH_SHARDED) {
                            return;
                        }
                        uint32_t M = (program_config.fuse_batch ? Volume(input_shape_a) / input_shape_a[-1]
                                                                : input_shape_a[-2]) /
                                     tt::constants::TILE_HEIGHT;
                        uint32_t N = input_shape_b[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        // No padding
                        if (M != per_core_M) {
                            return;
                        }

                        if (program_config.out_subblock_w != per_core_N && program_config.out_subblock_h != 1) {
                            return;
                        }
                    }
                } else {
                    if (memory_config_a.is_sharded()) {
                        if (!program_config.fuse_batch) {
                            return;
                        }
                        if (memory_config_a.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                            return;
                        }
                        if (memory_config_o.is_sharded()) {
                            if (memory_config_a.memory_layout != memory_config_o.memory_layout) {
                                return;
                            }
                        }
                        if (memory_config_a.shard_spec.value().orientation == ShardOrientation::ROW_MAJOR) {
                            return;
                        }
                        uint32_t M = (program_config.fuse_batch ? Volume(input_shape_a) / input_shape_a[-1]
                                                                : input_shape_a[-2]) /
                                     tt::constants::TILE_HEIGHT;
                        uint32_t K = input_shape_a[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = memory_config_a.shard_spec.value().shape;

                        if (tt::div_up(M, per_core_M) != memory_config_a.shard_spec.value().grid.num_cores()) {
                            return;
                        }
                        if (per_core_M != (shard_shape[0] / tt::constants::TILE_HEIGHT)) {
                            return;
                        }
                        if (K % program_config.in0_block_w != 0) {
                            return;
                        }
                        if (K != (shard_shape[1] / tt::constants::TILE_WIDTH)) {
                            return;
                        }
                    }
                    if (memory_config_o.is_sharded()) {
                        if (memory_config_o.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                            return;
                        }
                        uint32_t M = (program_config.fuse_batch ? Volume(input_shape_a) / input_shape_a[-1]
                                                                : input_shape_a[-2]) /
                                     tt::constants::TILE_HEIGHT;
                        uint32_t N = input_shape_b[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        if (N != per_core_N) {
                            return;
                        }
                        if (program_config.out_subblock_w != per_core_N && program_config.out_subblock_h != 1) {
                            return;
                        }
                    }
                }
                if (memory_config_b.memory_layout != TensorMemoryLayout::INTERLEAVED) {
                    return;
                }
                if ((input_shape_a[-1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w != 0) {
                    return;
                }
                if (program_config.per_core_M % program_config.out_subblock_h != 0) {
                    return;
                }
                if (program_config.per_core_N % program_config.out_subblock_w != 0) {
                    return;
                }
                matmul_op_type = MatmulOpTypes::MatmulMultiCoreReuseMultiCast1D;
                return;
            } else if constexpr (std::is_same_v<
                                     T,
                                     ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                if (memory_config_a.is_sharded()) {
                    auto tensor_a_memory_layout = memory_config_a.memory_layout;
                    uint32_t M = Volume(input_shape_a) / input_shape_a[-1] / tt::constants::TILE_HEIGHT;
                    uint32_t K = input_shape_a[-1] / tt::constants::TILE_WIDTH;
                    uint32_t N = input_shape_b[-1] / tt::constants::TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    auto shard_shape = memory_config_a.shard_spec.value().shape;

                    if (tensor_a_memory_layout != TensorMemoryLayout::BLOCK_SHARDED &&
                        tensor_a_memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                        return;
                    }

                    if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                        if (program_config.transpose_mcast) {
                            if (memory_config_a.shard_spec.value().orientation != ShardOrientation::COL_MAJOR) {
                                return;
                            }
                        } else {
                            if (memory_config_a.shard_spec.value().orientation != ShardOrientation::ROW_MAJOR) {
                                return;
                            }
                        }
                        if (memory_config_o.is_sharded()) {
                            if (memory_config_a.buffer_type != memory_config_o.buffer_type) {
                                return;
                            }
                            if (memory_config_a.memory_layout != memory_config_o.memory_layout) {
                                return;
                            }
                        }

                    } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                        if (program_config.transpose_mcast) {
                            return;
                        }
                        if (K != program_config.in0_block_w) {
                            return;
                        }
                        if (program_config.in0_block_w != (shard_shape[1] / tt::constants::TILE_WIDTH)) {
                            return;
                        }
                        if (memory_config_a.shard_spec.value().grid.bounding_box().start_coord.x !=
                            memory_config_a.shard_spec.value().grid.bounding_box().end_coord.x) {
                            return;
                        }
                    }
                    if (per_core_M != (shard_shape[0] / tt::constants::TILE_HEIGHT)) {
                        return;
                    }
                    if ((shard_shape[1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w != 0) {
                        return;
                    }
                }

                if (memory_config_b.is_sharded()) {
                    if (program_config.transpose_mcast) {
                        return;
                    }
                    auto tensor_b_memory_layout = memory_config_b.memory_layout;
                    if (tensor_b_memory_layout != TensorMemoryLayout::WIDTH_SHARDED) {
                        return;
                    }
                    if (memory_config_b.buffer_type != tt::tt_metal::BufferType::DRAM) {
                        if (program_config.per_core_N !=
                            (memory_config_b.shard_spec.value().shape[1] / tt::constants::TILE_WIDTH)) {
                            return;
                        }
                    }
                    if (memory_config_b.shard_spec.value().grid.bounding_box().start_coord.y !=
                        memory_config_b.shard_spec.value().grid.bounding_box().end_coord.y) {
                        return;
                    }
                }

                if (memory_config_o.is_sharded()) {
                    if (memory_config_o.memory_layout != TensorMemoryLayout::BLOCK_SHARDED) {
                        return;
                    }
                    uint32_t M = Volume(input_shape_a) / input_shape_a[-1] / tt::constants::TILE_HEIGHT;
                    uint32_t N = input_shape_b[-1] / tt::constants::TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    if (program_config.out_subblock_w != per_core_N && program_config.out_subblock_h != 1) {
                        return;
                    }
                }
                if ((input_shape_a[-1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w != 0) {
                    return;
                }
                if (program_config.per_core_M % program_config.out_subblock_h != 0) {
                    return;
                }
                if (program_config.per_core_N % program_config.out_subblock_w != 0) {
                    return;
                }
                matmul_op_type = MatmulOpTypes::MatmulMultiCoreReuseMultiCast;
                return;
            } else {
                std::cout << "Currently not supported" << std::endl;
                return;
            }
        },
        program_config);
    return matmul_op_type;
}
