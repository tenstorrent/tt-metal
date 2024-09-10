#include "unary_constraints.hpp"

namespace ttnn::operations::unary {

bool UnaryOpConstraintsBuilder::is_supported_dtype(DataType input_datatype, DataType output_datatype, UnaryOpType op_type) const
{
    switch (op_type) {
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FLOOR:
        case UnaryOpType::CEIL:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::RIGHT_SHIFT:
            break;
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
            if (input_datatype != DataType::INT32)
            {
                return false;
            }
            if (output_datatype != DataType::INT32)
            {
                return false;
            }
            break;
        case UnaryOpType::FMOD:
            if (input_datatype != DataType::BFLOAT16)
            {
                return false;
            }
            if (output_datatype != DataType::BFLOAT16)
            {
                return false;
            }
            break;
        default:
            return false;
    }
    return true;
}

std::vector<OpConstraint> UnaryOpConstraintsBuilder::build_constraints()
{
    if (can_build_constraints() == false)
    {
        throw std::runtime_error("Cannot build constraints, missing required parameters");
    }

    // reducing search space
    // data types are required
    static constexpr std::array<Layout, 2> tile_layouts = {Layout::ROW_MAJOR, Layout::TILE};
    // Only two for now. TODO: add other storage types.
    static constexpr std::array<StorageType, 2> storage_types = {StorageType::OWNED, StorageType::DEVICE};

    std::vector<OpConstraint> constraints;

    // Currently we are only looking at Unary for one input.
    for (const auto& tile_layout_a : tile_layouts)
    {
        for (const auto& storage_type_a : storage_types)
        {
            for (const auto& tile_layout_o : tile_layouts)
            {
                for (const auto& storage_type_o : storage_types)
                {
                    const auto constraint = OpConstraint(
                        data_type_a.value(),
                        tile_layout_a,
                        storage_type_a,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        data_type_o.value(),
                        tile_layout_o,
                        storage_type_o
                    );
                    if (is_valid_external_constraint(constraint)
                        && is_valid_op_constraint(constraint))
                    {
                        constraints.emplace_back(constraint);
                    }
                }
            }
        }
    }
    return std::move(constraints);
}

bool UnaryOpConstraintsBuilder::is_valid_op_constraint(const OpConstraint& constraint) const
{
    const tt::tt_metal::Layout c_tile_layout_a = constraint.getTileLayoutA().value();
    const tt::tt_metal::DataType data_type_a = constraint.getDataTypeA().value();
    tt::tt_metal::DataType data_type_o = constraint.getDataTypeO().value();
    if (!is_tensor_valid(memory_config_a, shape_a, c_tile_layout_a, data_type_a))
    {
        return false;
    }
    if (!is_supported_dtype(data_type_a, data_type_o, op_type))
    {
        return false;
    }
    if (storage_type_a != StorageType::DEVICE)
    {
        return false;
    }
    return true;
}

std::unique_ptr<UnaryOpConstraintsBuilder> UnaryOpConstraintsFactory::Make(
    const ttnn::operations::unary::UnaryOpType& _op_type,
    const tt::ARCH& arch,
    const ttnn::Shape& input_shape_a,
    const tt::tt_metal::MemoryConfig& memory_config_a,
    const ttnn::Shape& input_shape_o,
    const tt::tt_metal::MemoryConfig& memory_config_o,
    std::optional<const ttnn::Shape>& input_shape_b,
    std::optional<const tt::tt_metal::MemoryConfig>& memory_config_b)
{
    auto Unary_op_type = GetUnaryOpType(_op_type, arch, input_shape_a, memory_config_a, memory_config_o);
    switch (Unary_op_type) {
        case UnaryOpTypes::Unary:
            return std::make_unique<UnaryConstraintsBuilder>(_op_type, arch, input_shape_a, memory_config_a, input_shape_o, memory_config_o);
        case UnaryOpTypes::UnarySharded:
            return std::make_unique<UnaryShardedConstraintsBuilder>(_op_type, arch, input_shape_a, memory_config_a, input_shape_o, memory_config_o);

        default:
            return nullptr;
    }
};

bool UnaryOpConstraintsFactory::is_supported_arch(tt::ARCH arch, UnaryOpType op_type)
{
    switch (op_type) {
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FLOOR:
        case UnaryOpType::CEIL:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::RIGHT_SHIFT:
            if (arch == tt::ARCH::GRAYSKULL)
            {
                return false;
            }
            break;
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
            if (arch == tt::ARCH::GRAYSKULL)
            {
                return false;
            }
            break;
        case UnaryOpType::FMOD:
            if (arch == tt::ARCH::GRAYSKULL)
            {
                return false;
            }
            break;
        default:
            return false;
    }
    return true;
}

UnaryOpTypes UnaryOpConstraintsFactory::GetUnaryOpType(
        const ttnn::operations::unary::UnaryOpType& _op_type,
        const tt::ARCH& arch,
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const tt::tt_metal::MemoryConfig& memory_config_o)
{
    if (!is_supported_arch(arch, _op_type))
    {
        return UnaryOpTypes::NotSupported;
    }

    if (memory_config_a.memory_layout != memory_config_o.memory_layout)
    {
        return UnaryOpTypes::NotSupported;
    }

    if (!memory_config_a.is_sharded()) {
        if (memory_config_a.memory_layout != TensorMemoryLayout::INTERLEAVED)
        {
            return UnaryOpTypes::NotSupported;
        }
        return UnaryOpTypes::Unary;
    }
    else
    {
        return UnaryOpTypes::UnarySharded;
    }
}

}
