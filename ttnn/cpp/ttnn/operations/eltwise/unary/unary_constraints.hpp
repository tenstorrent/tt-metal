#pragma once

#include <memory>
#include <optional>
#include <tuple>

#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/common/op_constraints.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::unary {

enum class UnaryOpTypes { Unary, UnarySharded, NotSupported };

class UnaryOpConstraintsBuilder : public OpConstraintsBuilder {
   protected:
    const ttnn::operations::unary::UnaryOpType op_type;
    const tt::ARCH arch;
    const ttnn::Shape shape_a;
    const tt::tt_metal::MemoryConfig memory_config_a;
    const ttnn::Shape shape_o;
    const tt::tt_metal::MemoryConfig memory_config_o;

    UnaryOpConstraintsBuilder(
        const ttnn::operations::unary::UnaryOpType& _op_type,
        const tt::ARCH& _arch,
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_o,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        op_type(_op_type),
        arch(_arch),
        shape_a(_shape_a),
        memory_config_a(_memory_config_a),
        shape_o(_shape_o),
        memory_config_o(_memory_config_o) {}

    bool is_supported_dtype(DataType input_datatype, DataType output_datatype, UnaryOpType op_type) const;

    // op specific constraints
    virtual bool is_valid_op_constraint(const OpConstraint& constraint) const override;

   public:
    virtual ~UnaryOpConstraintsBuilder() = default;

    virtual bool can_build_constraints() const override;

    std::vector<OpConstraint> build_constraints() override;
};

class UnaryConstraintsBuilder : public UnaryOpConstraintsBuilder {
   public:
    UnaryConstraintsBuilder(
        const ttnn::operations::unary::UnaryOpType& _op_type,
        const tt::ARCH& _arch,
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_o,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        UnaryOpConstraintsBuilder(_op_type, _arch, _shape_a, _memory_config_a, _shape_o, _memory_config_o) {
        std::cout << "UnaryConstraintsBuilder" << std::endl;
    }
    virtual ~UnaryConstraintsBuilder() = default;

    virtual std::string get_op_name() const override { return "Unary"; }
};

class UnaryShardedConstraintsBuilder : public UnaryOpConstraintsBuilder {
   public:
    UnaryShardedConstraintsBuilder(
        const ttnn::operations::unary::UnaryOpType& _op_type,
        const tt::ARCH& _arch,
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_o,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        UnaryOpConstraintsBuilder(_op_type, _arch, _shape_a, _memory_config_a, _shape_o, _memory_config_o) {
        std::cout << "UnaryShardedConstraintsBuilder" << std::endl;
    }
    virtual ~UnaryShardedConstraintsBuilder() = default;

    virtual std::string get_op_name() const override { return "UnarySharded"; }
};

class UnaryOpConstraintsFactory {
   public:
    UnaryOpConstraintsFactory() = delete;
    static std::unique_ptr<UnaryOpConstraintsBuilder> Make(
        const ttnn::operations::unary::UnaryOpType& _op_type,
        const tt::ARCH& arch,
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_o,
        const tt::tt_metal::MemoryConfig& memory_config_o,
        const CoreCoord& chip_size);

    static bool is_supported_arch(tt::ARCH arch, UnaryOpType op_type);

    static UnaryOpTypes GetUnaryOpType(
        const ttnn::operations::unary::UnaryOpType& _op_type,
        const tt::ARCH& arch,
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const tt::tt_metal::MemoryConfig& memory_config_o);
};

}  // namespace ttnn::operations::unary
