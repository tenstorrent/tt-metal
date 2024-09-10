#include "softmax_constraints.hpp"

std::vector<OpConstraint> SoftmaxOpConstraintsBuilder::build_constraints() {
    if (can_build_constraints() == false) {
        throw std::runtime_error("Cannot build constraints, missing required parameters");
    }

    // reducing search space
    // data types are required
    static constexpr std::array<Layout, 2> tile_layouts = {Layout::ROW_MAJOR, Layout::TILE};
    // Only two for now. TODO: add other storage types.
    static constexpr std::array<StorageType, 2> storage_types = {StorageType::OWNED, StorageType::DEVICE};

    std::vector<OpConstraint> constraints;

    // Currently we are only looking at softmax for one input.
    for (const auto& tile_layout_a : tile_layouts) {
        for (const auto& storage_type_a : storage_types) {
            for (const auto& tile_layout_o : tile_layouts) {
                for (const auto& storage_type_o : storage_types) {
                    const auto constraint = OpConstraint(
                        data_type_a.value(),
                        tile_layout_a,
                        storage_type_a,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
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
    return std::move(constraints);
}

bool SoftmaxConstraintsBuilder::is_valid_op_constraint(const OpConstraint& constraint) const {
    const tt::tt_metal::DataType data_type_a = constraint.getDataTypeA().value();
    const tt::tt_metal::Layout c_tile_layout_a = constraint.getTileLayoutA().value();
    if (!is_tensor_valid(memory_config_a, shape_a, c_tile_layout_a, data_type_a)) {
        return false;
    }
    // made-up constraint - tiles only
    if (c_tile_layout_a != tt::tt_metal::Layout::TILE) {
        return false;
    }
    if (!(data_type_a == tt::tt_metal::DataType::FLOAT32 || data_type_a == tt::tt_metal::DataType::BFLOAT16 ||
          data_type_a == tt::tt_metal::DataType::BFLOAT8_B)) {
        return false;
    }

    return true;
}

std::unique_ptr<SoftmaxOpConstraintsBuilder> SoftmaxOpConstraintsFactory::Make(
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::Shape& input_shape_o,
    const tt::tt_metal::MemoryConfig& memory_config_o,
    std::optional<const ttnn::Shape>& input_shape_b,
    std::optional<const tt::tt_metal::MemoryConfig>& memory_config_b) {
    auto Softmax_op_type = GetSoftmaxOpType(input_shape_a, memory_config_a, input_shape_b, memory_config_b);
    switch (Softmax_op_type) {
        case SoftmaxOpTypes::Softmax:
            return std::make_unique<SoftmaxConstraintsBuilder>(
                input_shape_a, memory_config_a, input_shape_o, memory_config_o);
        case SoftmaxOpTypes::SoftmaxInPlace:
        case SoftmaxOpTypes::ScaleMaskSoftmaxInPlace:              // not implemented yet
        case SoftmaxOpTypes::ScaleCausalMaskHwDimsSoftmaxInPlace:  // not implemented yet
        case SoftmaxOpTypes::ScaleMaskSoftmax:                     // not implemented yet
        default: return nullptr;
    }
};

SoftmaxOpTypes SoftmaxOpConstraintsFactory::GetSoftmaxOpType(
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    std::optional<const ttnn::Shape>& input_shape_b,
    std::optional<const tt::tt_metal::MemoryConfig>& memory_config_b) {
    if (input_shape_b.has_value() || memory_config_b.has_value()) {
        std::cout << "SoftmaxOpTypes::NotSupported" << std::endl;
        return SoftmaxOpTypes::NotSupported;
    }
    return SoftmaxOpTypes::Softmax;
}
