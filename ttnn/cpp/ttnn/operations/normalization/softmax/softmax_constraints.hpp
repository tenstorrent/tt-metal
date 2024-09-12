#pragma once

#include <memory>
#include <optional>
#include <tuple>

#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/common/op_constraints.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

enum class SoftmaxOpTypes {
    SoftmaxInPlace,
    ScaleMaskSoftmaxInPlace,
    ScaleCausalMaskHwDimsSoftmaxInPlace,
    Softmax,
    ScaleMaskSoftmax,
    NotSupported
};

class SoftmaxOpConstraintsBuilder : public OpConstraintsBuilder {
   protected:
    const ttnn::Shape shape_a;
    const tt::tt_metal::MemoryConfig memory_config_a;
    const ttnn::Shape shape_o;
    const tt::tt_metal::MemoryConfig memory_config_o;

    SoftmaxOpConstraintsBuilder(
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_o,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        shape_a(_shape_a), memory_config_a(_memory_config_a), shape_o(_shape_o), memory_config_o(_memory_config_o) {}

   public:
    virtual ~SoftmaxOpConstraintsBuilder() = default;

    virtual std::vector<OpConstraint> build_constraints() override;
};

class SoftmaxConstraintsBuilder : public SoftmaxOpConstraintsBuilder {
   public:
    // ElementWiseMultiCoreConstraintsBuilder() = default;
    SoftmaxConstraintsBuilder(
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_o,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        SoftmaxOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_o, _memory_config_o) {
        std::cout << "SoftmaxConstraintsBuilder" << std::endl;
    }
    virtual ~SoftmaxConstraintsBuilder() = default;

    virtual std::string get_op_name() const override { return "Softmax"; }

    virtual bool can_build_constraints() const override;

   protected:
    virtual bool is_valid_op_constraint(const OpConstraint& constraint) const override;
};
class SoftmaxOpConstraintsFactory {
   public:
    SoftmaxOpConstraintsFactory() = delete;
    static std::unique_ptr<SoftmaxOpConstraintsBuilder> Make(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_o,
        const tt::tt_metal::MemoryConfig& memory_config_o,
        const std::optional<const ttnn::Shape>& input_shape_b = std::nullopt,
        const std::optional<const tt::tt_metal::MemoryConfig>& memory_config_b = std::nullopt);

    static SoftmaxOpTypes GetSoftmaxOpType(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const std::optional<const ttnn::Shape>& input_shape_b = std::nullopt,
        const std::optional<const tt::tt_metal::MemoryConfig>& memory_config_b = std::nullopt);
};
