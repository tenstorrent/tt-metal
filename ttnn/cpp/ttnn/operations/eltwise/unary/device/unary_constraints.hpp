#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
// input shapes and shard shapes are runtime thing, we cannot return arbitrary count of shapes

namespace ttnn::operations::unary {

// Currently we are only looking at Unary for one input.
using UnaryOpConstraint = std::tuple<
    // a
    tt::tt_metal::DataType,
    tt::tt_metal::Layout,
    tt::tt_metal::StorageType,
    // output
    tt::tt_metal::DataType,
    tt::tt_metal::Layout,
    tt::tt_metal::StorageType
>;

class UnaryOpConstraintsBuilder {
    protected:
        const ttnn::operations::unary::UnaryOpType op_type;
        const tt::ARCH arch;
        const ttnn::Shape shape_a;
        const tt::tt_metal::MemoryConfig memory_config_a;
        const ttnn::Shape shape_o;
        const tt::tt_metal::MemoryConfig memory_config_o;

        std::optional<tt::tt_metal::DataType> data_type_a;  // required
        std::optional<tt::tt_metal::Layout> tile_layout_a;
        std::optional<tt::tt_metal::StorageType> storage_type_a;


        std::optional<tt::tt_metal::DataType> data_type_o; // required
        std::optional<tt::tt_metal::Layout> tile_layout_o;
        std::optional<tt::tt_metal::StorageType> storage_type_o;


        UnaryOpConstraintsBuilder(const ttnn::operations::unary::UnaryOpType& _op_type,
                                  const tt::ARCH& _arch,
                                  const ttnn::Shape& _shape_a,
                                  const tt::tt_metal::MemoryConfig& _memory_config_a,
                                  const ttnn::Shape& _shape_o,
                                  const tt::tt_metal::MemoryConfig& _memory_config_o) :
                                  op_type(_op_type), arch(_arch),
                                  shape_a(_shape_a), memory_config_a(_memory_config_a),
                                  shape_o(_shape_o), memory_config_o(_memory_config_o) {}

        // check if required parameters are set
        bool can_build_constraints() const
        {
            return (data_type_a.has_value() && data_type_o.has_value());
        }

        bool is_supported_dtype(DataType input_datatype, DataType output_datatype, UnaryOpType op_type) const
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

        // check if it is possible to build constraints with all set parameters
        bool is_valid_external_constraint(const UnaryOpConstraint& constraint) const {
            if (data_type_a.has_value() && std::get<0>(constraint) != data_type_a.value()) {
                return false;
            }
            if (tile_layout_a.has_value() && std::get<1>(constraint) != tile_layout_a.value()) {
                return false;
            }
            if (storage_type_a.has_value() && std::get<2>(constraint) != storage_type_a.value()) {
                return false;
            }
            if (data_type_o.has_value() && std::get<3>(constraint) != data_type_o.value()) {
                return false;
            }
            if (tile_layout_o.has_value() && std::get<4>(constraint) != tile_layout_o.value()) {
                return false;
            }
            if (storage_type_o.has_value() && std::get<5>(constraint) != storage_type_o.value()) {
                return false;
            }
            return true;
        }

        // op specific constraints
        virtual bool is_valid_op_constraint(const UnaryOpConstraint& constraint) const = 0;

    public:
        virtual ~UnaryOpConstraintsBuilder() = default;

        virtual std::string get_op_name() const = 0;

        std::vector<UnaryOpConstraint> build_constraints()
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

            std::vector<UnaryOpConstraint> constraints;

            // Currently we are only looking at Unary for one input.
            for (const auto& tile_layout_a : tile_layouts)
            {
                for (const auto& storage_type_a : storage_types)
                {
                    for (const auto& tile_layout_o : tile_layouts)
                    {
                        for (const auto& storage_type_o : storage_types)
                        {
                            const auto constraint = std::make_tuple(
                                data_type_a.value(),
                                tile_layout_a,
                                storage_type_a,
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

        // Setters for parameter a
        UnaryOpConstraintsBuilder& setDataTypeA(tt::tt_metal::DataType dataType) {
            data_type_a = dataType;
            return *this;
        }

        UnaryOpConstraintsBuilder& setTileLayoutA(tt::tt_metal::Layout tileLayout) {
            tile_layout_a = tileLayout;
            return *this;
        }

        UnaryOpConstraintsBuilder& setStorageTypeA(tt::tt_metal::StorageType storageType)
        {
            storage_type_a = storageType;
            return *this;
        }

        // Setters for parameter output
        UnaryOpConstraintsBuilder& setDataTypeO(tt::tt_metal::DataType dataType) {
            data_type_o = dataType;
            return *this;
        }

        UnaryOpConstraintsBuilder& setTileLayoutO(tt::tt_metal::Layout tileLayout) {
            tile_layout_o = tileLayout;
            return *this;
        }

        UnaryOpConstraintsBuilder& setStorageTypeO(tt::tt_metal::StorageType storageType)
        {
            storage_type_o = storageType;
            return *this;
        }
};

enum class UnaryOpTypes
{
    Unary,
    UnarySharded,
    NotSupported
};

class UnaryConstraintsBuilder : public UnaryOpConstraintsBuilder
{
public:
    // ElementWiseMultiCoreConstraintsBuilder() = default;
    UnaryConstraintsBuilder(const ttnn::operations::unary::UnaryOpType& _op_type,
                            const tt::ARCH& _arch,
                            const ttnn::Shape& _shape_a,
                            const tt::tt_metal::MemoryConfig& _memory_config_a,
                            const ttnn::Shape& _shape_o,
                            const tt::tt_metal::MemoryConfig& _memory_config_o)
                              : UnaryOpConstraintsBuilder(_op_type, _arch, _shape_a, _memory_config_a, _shape_o, _memory_config_o)
    {
        std::cout << "UnaryConstraintsBuilder" << std::endl;
    }
    virtual ~UnaryConstraintsBuilder() = default;

    virtual std::string get_op_name() const override {
        return "Unary";
    }

protected:
    virtual bool is_valid_op_constraint(const UnaryOpConstraint& constraint) const override {
        tt::tt_metal::DataType data_type_a = std::get<0>(constraint);
        tt::tt_metal::DataType data_type_o = std::get<3>(constraint);
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
};

class UnaryShardedConstraintsBuilder : public UnaryOpConstraintsBuilder
{
public:
    UnaryShardedConstraintsBuilder(const ttnn::operations::unary::UnaryOpType& _op_type,
                            const tt::ARCH& _arch,
                            const ttnn::Shape& _shape_a,
                            const tt::tt_metal::MemoryConfig& _memory_config_a,
                            const ttnn::Shape& _shape_o,
                            const tt::tt_metal::MemoryConfig& _memory_config_o)
                              : UnaryOpConstraintsBuilder(_op_type, _arch, _shape_a, _memory_config_a, _shape_o, _memory_config_o)
    {
        std::cout << "UnaryShardedConstraintsBuilder" << std::endl;
    }
    virtual ~UnaryShardedConstraintsBuilder() = default;

    virtual std::string get_op_name() const override {
        return "UnarySharded";
    }

protected:
    virtual bool is_valid_op_constraint(const UnaryOpConstraint& constraint) const override {
        tt::tt_metal::DataType data_type_a = std::get<0>(constraint);
        tt::tt_metal::DataType data_type_o = std::get<3>(constraint);
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
};

class UnaryOpConstraintsFactory
{
    public:
    UnaryOpConstraintsFactory() = delete;
    static std::unique_ptr<UnaryOpConstraintsBuilder> Make(
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

    static bool is_supported_arch(tt::ARCH arch, UnaryOpType op_type)
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

    static UnaryOpTypes GetUnaryOpType(
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
};

}
