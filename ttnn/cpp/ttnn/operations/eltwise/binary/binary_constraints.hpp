#pragma once

#include <algorithm>
#include <memory>
#include <optional>

#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/common/op_constraints.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

enum class EltwiseOpTypes {
    ElementWiseMultiCore,
    BroadcastWidthMultiCore,
    BroadcastHeightMultiCore,
    BroadcastHeightAndWidthMultiCore,
    BroadcastHeightMultiCoreSharded,
    BroadcastHeightMultiCoreShardedOptimized,
    NotSupported
};

class EltwiseOpConstraintsBuilder : public OpConstraintsBuilder {
   protected:
    const ttnn::Shape shape_a;
    const tt::tt_metal::MemoryConfig memory_config_a;
    const ttnn::Shape shape_b;
    const tt::tt_metal::MemoryConfig memory_config_b;
    const tt::tt_metal::MemoryConfig memory_config_o;

    EltwiseOpConstraintsBuilder(
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_b,
        const tt::tt_metal::MemoryConfig& _memory_config_b,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        shape_a(_shape_a),
        memory_config_a(_memory_config_a),
        shape_b(_shape_b),
        memory_config_b(_memory_config_b),
        memory_config_o(_memory_config_o) {}

   public:
    virtual ~EltwiseOpConstraintsBuilder() = default;

    virtual std::vector<OpConstraint> build_constraints() override;

    virtual bool is_valid_op_constraint(const OpConstraint& constraint) const override;
};

class ElementWiseMultiCoreConstraintsBuilder : public EltwiseOpConstraintsBuilder {
   public:
    // ElementWiseMultiCoreConstraintsBuilder() = default;
    ElementWiseMultiCoreConstraintsBuilder(
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_b,
        const tt::tt_metal::MemoryConfig& _memory_config_b,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        EltwiseOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_b, _memory_config_b, _memory_config_o) {
        std::cout << "ElementWiseMultiCoreConstraintsBuilder" << std::endl;
    }

    std::string get_op_name() const override { return "ElementWiseMultiCore"; }

    static bool check_input_parameters(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b);
};

class BroadcastWidthMultiCoreConstraintsBuilder : public EltwiseOpConstraintsBuilder {
   public:
    BroadcastWidthMultiCoreConstraintsBuilder(
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_b,
        const tt::tt_metal::MemoryConfig& _memory_config_b,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        EltwiseOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_b, _memory_config_b, _memory_config_o) {
        std::cout << "BroadcastWidthMultiCoreConstraintsBuilder" << std::endl;
    }

    virtual std::string get_op_name() const override { return "BroadcastWidthMultiCoreConstraintsBuilder"; }

    static bool check_input_parameters(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b);
};

class EltwiseOpConstraintsFactory {
   public:
    EltwiseOpConstraintsFactory() = delete;
    static std::unique_ptr<EltwiseOpConstraintsBuilder> Make(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b,
        const tt::tt_metal::MemoryConfig& memory_config_o,
        const CoreCoord& chip_grid);

    static EltwiseOpTypes GetEltwiseOpType(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b,
        const tt::tt_metal::MemoryConfig& memory_config_o);
};
