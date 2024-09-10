#pragma once

#include <algorithm>
#include <memory>
#include <optional>

#include "common/core_coord.h"
#include "impl/buffers/buffer_constants.hpp"
#include "tt_metal/common/core_coord.h"
#include "ttnn/common/op_constraints.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

enum class MatmulOpTypes {
    MatmulMultiCore,
    MatmulMultiCoreNonOptimizedReuse,
    MatmulMultiCoreReuse,
    MatmulMultiCoreReuseMultiCast,
    MatmulMultiCoreReuseMultiCast1D,
    MatmulMultiCoreReuseMultiCastDRAMSharded,
    NotSupported
};

class MatmulOpConstraintsBuilder : public OpConstraintsBuilder {
   protected:
    const ttnn::Shape shape_a;
    const tt::tt_metal::MemoryConfig memory_config_a;
    const ttnn::Shape shape_b;
    const tt::tt_metal::MemoryConfig memory_config_b;
    const tt::tt_metal::MemoryConfig memory_config_o;

    MatmulOpConstraintsBuilder(
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
    virtual ~MatmulOpConstraintsBuilder() = default;

    std::vector<OpConstraint> build_constraints() override;

    virtual bool is_valid_op_constraint(const OpConstraint& constraint) const override;
};

class MatmulMultiCoreReuseMultiCastConstraintsBuilder : public MatmulOpConstraintsBuilder {
   public:
    MatmulMultiCoreReuseMultiCastConstraintsBuilder(
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_b,
        const tt::tt_metal::MemoryConfig& _memory_config_b,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        MatmulOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_b, _memory_config_b, _memory_config_o) {
        std::cout << "MatmulMultiCoreReuseMultiCastConstraintsBuilder" << std::endl;
    }
    virtual ~MatmulMultiCoreReuseMultiCastConstraintsBuilder() = default;

    virtual std::string get_op_name() const override { return "MatmulMultiCoreReuseMultiCast"; }
};

class MatmulMultiCoreReuseMultiCast1DConstraintsBuilder : public MatmulOpConstraintsBuilder {
   public:
    // BroadcastWidthMultiCoreConstraintsBuilder() = default;
    MatmulMultiCoreReuseMultiCast1DConstraintsBuilder(
        const ttnn::Shape& _shape_a,
        const tt::tt_metal::MemoryConfig& _memory_config_a,
        const ttnn::Shape& _shape_b,
        const tt::tt_metal::MemoryConfig& _memory_config_b,
        const tt::tt_metal::MemoryConfig& _memory_config_o) :
        MatmulOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_b, _memory_config_b, _memory_config_o) {
        std::cout << "MatmulMultiCoreReuseMultiCast1DConstraintsBuilder" << std::endl;
    }
    virtual ~MatmulMultiCoreReuseMultiCast1DConstraintsBuilder() = default;

    virtual std::string get_op_name() const override { return "MatmulMultiCoreReuseMultiCast1D"; }
};

class MatmulOpConstraintsFactory {
   public:
    MatmulOpConstraintsFactory() = delete;
    static std::unique_ptr<MatmulOpConstraintsBuilder> Make(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        tt::tt_metal::MemoryConfig& memory_config_b,
        tt::tt_metal::MemoryConfig& memory_config_o,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config);

    static const uint32_t Volume(const ttnn::Shape& shape);

    static MatmulOpTypes GetMatmulOpType(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b,
        const tt::tt_metal::MemoryConfig& memory_config_o,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config);
};
