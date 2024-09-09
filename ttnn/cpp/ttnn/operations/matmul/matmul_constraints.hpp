#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include "common/core_coord.h"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/common/core_coord.h"

using MatmulOpConstraint = std::tuple<
    // a
    tt::tt_metal::DataType,
    tt::tt_metal::Layout,
    // b
    tt::tt_metal::DataType,
    tt::tt_metal::Layout,
    // output
    tt::tt_metal::DataType,
    tt::tt_metal::Layout
>;

class MatmulOpConstraintsBuilder {
    protected:
        const ttnn::Shape shape_a;
        const tt::tt_metal::MemoryConfig memory_config_a;
        const ttnn::Shape shape_b;
        const tt::tt_metal::MemoryConfig memory_config_b;
        const tt::tt_metal::MemoryConfig memory_config_o;


        std::optional<tt::tt_metal::DataType> data_type_a;  // required
        std::optional<tt::tt_metal::Layout> tile_layout_a;

        std::optional<tt::tt_metal::DataType> data_type_b;  // required
        std::optional<tt::tt_metal::Layout> tile_layout_b;

        std::optional<tt::tt_metal::DataType> data_type_o;  // required
        std::optional<tt::tt_metal::Layout> tile_layout_o;

        MatmulOpConstraintsBuilder(const ttnn::Shape& _shape_a,
                                    const tt::tt_metal::MemoryConfig& _memory_config_a,
                                    const ttnn::Shape& _shape_b,
                                    const tt::tt_metal::MemoryConfig& _memory_config_b,
                                    const tt::tt_metal::MemoryConfig& _memory_config_o) :
                                    shape_a(_shape_a), memory_config_a(_memory_config_a),
                                    shape_b(_shape_b), memory_config_b(_memory_config_b),
                                    memory_config_o(_memory_config_o) {}

        // check if required parameters are set
        bool can_build_constraints() const
        {
            return data_type_a.has_value() && data_type_b.has_value() && data_type_o.has_value();
        }

        // check if it is possible to build constraints with all set parameters
        bool is_valid_external_constraint(const MatmulOpConstraint& constraint) const {
            if (data_type_a.has_value() && std::get<0>(constraint) != data_type_a.value()) {
                return false;
            }
            if (tile_layout_a.has_value() && std::get<1>(constraint) != tile_layout_a.value()) {
                return false;
            }
            if (data_type_b.has_value() && std::get<2>(constraint) != data_type_o.value()) {
                return false;
            }
            if (tile_layout_b.has_value() && std::get<3>(constraint) != tile_layout_o.value()) {
                return false;
            }
            if (data_type_o.has_value() && std::get<4>(constraint) != data_type_o.value()) {
                return false;
            }
            if (tile_layout_o.has_value() && std::get<5>(constraint) != tile_layout_o.value()) {
                return false;
            }
            return true;
        }

        // op specific constraints
        virtual bool is_valid_op_constraint(const MatmulOpConstraint& constraint) const = 0;

    public:
        virtual ~MatmulOpConstraintsBuilder() = default;

        virtual std::string get_op_name() const = 0;

        std::vector<MatmulOpConstraint> build_constraints()
        {
            if (can_build_constraints() == false)
            {
                throw std::runtime_error("Cannot build constraints, missing required parameters");
            }

             // reducing search space
            // data types are required
            static constexpr std::array<Layout, 2> tile_layouts = {Layout::ROW_MAJOR, Layout::TILE};

            std::vector<MatmulOpConstraint> constraints;

            // Currently we are only looking at softmax for one input.
            for (const auto& tile_layout_a : tile_layouts)
            {
                for (const auto& tile_layout_b : tile_layouts)
                {
                    for (const auto& tile_layout_o : tile_layouts)
                    {
                        const auto constraint = std::make_tuple(
                            data_type_a.value(),
                            tile_layout_a,
                            data_type_b.value(),
                            tile_layout_b,
                            data_type_o.value(),
                            tile_layout_o
                        );
                        if (is_valid_external_constraint(constraint)
                            && is_valid_op_constraint(constraint))
                        {
                            constraints.emplace_back(constraint);
                        }
                    }
                }
            }

            return std::move(constraints);
        }

        // Setters for parameter a
        MatmulOpConstraintsBuilder& setDataTypeA(tt::tt_metal::DataType dataType) {
            data_type_a = dataType;
            return *this;
        }

        MatmulOpConstraintsBuilder& setTileLayoutA(tt::tt_metal::Layout tileLayout) {
            tile_layout_a = tileLayout;
            return *this;
        }

        // Setters for parameter b
        MatmulOpConstraintsBuilder& setDataTypeB(tt::tt_metal::DataType dataType) {
            data_type_b = dataType;
            return *this;
        }

        MatmulOpConstraintsBuilder& setTileLayoutB(tt::tt_metal::Layout tileLayout) {
            tile_layout_b = tileLayout;
            return *this;
        }

        // Setters for parameter output
        MatmulOpConstraintsBuilder& setDataTypeO(tt::tt_metal::DataType dataType) {
            data_type_o = dataType;
            return *this;
        }

        MatmulOpConstraintsBuilder& setTileLayoutO(tt::tt_metal::Layout tileLayout) {
            tile_layout_o = tileLayout;
            return *this;
        }
};

enum class MatmulOpTypes
{
    MatmulMultiCore,
    MatmulMultiCoreNonOptimizedReuse,
    MatmulMultiCoreReuse,
    MatmulMultiCoreReuseMultiCast,
    MatmulMultiCoreReuseMultiCast1D,
    MatmulMultiCoreReuseMultiCastDRAMSharded,
    NotSupported
};

class MatmulMultiCoreReuseMultiCastConstraintsBuilder : public MatmulOpConstraintsBuilder
{
public:
    MatmulMultiCoreReuseMultiCastConstraintsBuilder(const ttnn::Shape& _shape_a,
                                                    const tt::tt_metal::MemoryConfig& _memory_config_a,
                                                    const ttnn::Shape& _shape_b,
                                                    const tt::tt_metal::MemoryConfig& _memory_config_b,
                                                    const tt::tt_metal::MemoryConfig& _memory_config_o)
    : MatmulOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_b, _memory_config_b, _memory_config_o)
    {
        std::cout << "MatmulMultiCoreReuseMultiCastConstraintsBuilder" << std::endl;
    }
    virtual ~MatmulMultiCoreReuseMultiCastConstraintsBuilder() = default;

    virtual std::string get_op_name() const override {
        return "MatmulMultiCoreReuseMultiCast";
    }

protected:
    virtual bool is_valid_op_constraint(const MatmulOpConstraint& constraint) const override {
        const tt::tt_metal::DataType data_type_a = std::get<0>(constraint);
        if (!is_floating_point(data_type_a))
        {
            return false;
        }
        return true;
    }
};

class MatmulMultiCoreReuseMultiCast1DConstraintsBuilder : public MatmulOpConstraintsBuilder
{
public:
    // BroadcastWidthMultiCoreConstraintsBuilder() = default;
    MatmulMultiCoreReuseMultiCast1DConstraintsBuilder(const ttnn::Shape& _shape_a,
                                                    const tt::tt_metal::MemoryConfig& _memory_config_a,
                                                    const ttnn::Shape& _shape_b,
                                                    const tt::tt_metal::MemoryConfig& _memory_config_b,
                                                    const tt::tt_metal::MemoryConfig& _memory_config_o)
    : MatmulOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_b, _memory_config_b, _memory_config_o)
    {
        std::cout << "MatmulMultiCoreReuseMultiCast1DConstraintsBuilder" << std::endl;
    }
    virtual ~MatmulMultiCoreReuseMultiCast1DConstraintsBuilder() = default;

    virtual std::string get_op_name() const override {
        return "MatmulMultiCoreReuseMultiCast1D";
    }

protected:
    virtual bool is_valid_op_constraint(const MatmulOpConstraint& constraint) const override {
        const tt::tt_metal::DataType data_type_a = std::get<0>(constraint);
        if (!is_floating_point(data_type_a))
        {
            return false;
        }
        return true;
    }
};

class MatmulOpConstraintsFactory
{
    public:
    MatmulOpConstraintsFactory() = delete;
    static std::unique_ptr<MatmulOpConstraintsBuilder> Make(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        tt::tt_metal::MemoryConfig& memory_config_b,
        tt::tt_metal::MemoryConfig& memory_config_o,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config)
    {
        auto Matmul_op_type = GetMatmulOpType(input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o, program_config);
        switch (Matmul_op_type) {
            case MatmulOpTypes::MatmulMultiCoreReuseMultiCast:
                return std::make_unique<MatmulMultiCoreReuseMultiCastConstraintsBuilder>(input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
            case MatmulOpTypes::MatmulMultiCoreReuseMultiCast1D:
                return std::make_unique<MatmulMultiCoreReuseMultiCast1DConstraintsBuilder>(input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
            case MatmulOpTypes::MatmulMultiCore: // not implemented yet
            case MatmulOpTypes::MatmulMultiCoreNonOptimizedReuse: // not implemented yet
            case MatmulOpTypes::MatmulMultiCoreReuse: // not implemented yet
            case MatmulOpTypes::MatmulMultiCoreReuseMultiCastDRAMSharded: // not implemented yet
            default:
                return nullptr;
        }
    };

    static const uint32_t Volume(const ttnn::Shape& shape)
    {
        auto rank = shape.rank();
        auto volume = 1;
        for (auto index = 0; index < rank; index++) {
            volume *= shape.operator[](index);
        }
        return volume;
    }

    static MatmulOpTypes GetMatmulOpType(const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b,
        const tt::tt_metal::MemoryConfig& memory_config_o,
        const ttnn::operations::matmul::MatmulProgramConfig& program_config)
    {
        if (input_shape_a[-1] != input_shape_b[-2])
        {
            return MatmulOpTypes::NotSupported;
        }
        if (input_shape_a.rank() != input_shape_b.rank())
        {
            return MatmulOpTypes::NotSupported;
        }
        for (auto i = 0; i < input_shape_a.rank() - 2; i++) {
            if (input_shape_a[i] != input_shape_b[i])
            {
                return MatmulOpTypes::NotSupported;
            }
        }
        std::visit(
            [&](const auto& program_config) {
                using T = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<T, ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    if (program_config.mcast_in0)
                    {
                        if (memory_config_a.is_sharded())
                        {
                            if (!program_config.fuse_batch)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (memory_config_a.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (memory_config_o.is_sharded())
                            {
                                if (memory_config_a.memory_layout != memory_config_o.memory_layout)
                                {
                                    return MatmulOpTypes::NotSupported;
                                }
                            }
                            uint32_t M =
                                (program_config.fuse_batch ? Volume(input_shape_a) / input_shape_a[-1]
                                                        : input_shape_a[-2]) /
                                tt::constants::TILE_HEIGHT;
                            uint32_t K = input_shape_a[-1] / tt::constants::TILE_WIDTH;
                            uint32_t per_core_M = program_config.per_core_M;
                            auto shard_shape = memory_config_a.shard_spec.value().shape;

                            if (M != per_core_M)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (per_core_M != (shard_shape[0] / tt::constants::TILE_HEIGHT))
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (K % program_config.in0_block_w != 0)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if ((shard_shape[1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w != 0)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                        }
                        if (memory_config_o.is_sharded()) {
                            if (memory_config_o.memory_layout != TensorMemoryLayout::WIDTH_SHARDED)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            uint32_t M =
                                (program_config.fuse_batch ? Volume(input_shape_a) / input_shape_a[-1]
                                                        : input_shape_a[-2]) /
                                tt::constants::TILE_HEIGHT;
                            uint32_t N = input_shape_b[-1] / tt::constants::TILE_WIDTH;
                            uint32_t per_core_M = program_config.per_core_M;
                            uint32_t per_core_N = program_config.per_core_N;

                            // No padding
                            if (M != per_core_M)
                            {
                                return MatmulOpTypes::NotSupported;
                            }

                            if (program_config.out_subblock_w != per_core_N && program_config.out_subblock_h != 1)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                        }
                    }
                    else
                    {
                        if (memory_config_a.is_sharded())
                        {
                            if (!program_config.fuse_batch)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (memory_config_a.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (memory_config_o.is_sharded()) {
                                if (memory_config_a.memory_layout != memory_config_o.memory_layout)
                                {
                                    return MatmulOpTypes::NotSupported;
                                }
                            }
                            if (memory_config_a.shard_spec.value().orientation == ShardOrientation::ROW_MAJOR)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            uint32_t M =
                                (program_config.fuse_batch ? Volume(input_shape_a)/ input_shape_a[-1]
                                                        : input_shape_a[-2]) /
                                tt::constants::TILE_HEIGHT;
                            uint32_t K = input_shape_a[-1] / tt::constants::TILE_WIDTH;
                            uint32_t per_core_M = program_config.per_core_M;
                            auto shard_shape = memory_config_a.shard_spec.value().shape;

                            if (tt::div_up(M, per_core_M) != memory_config_a.shard_spec.value().grid.num_cores())
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (per_core_M != (shard_shape[0] / tt::constants::TILE_HEIGHT))
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (K % program_config.in0_block_w != 0)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (K != (shard_shape[1] / tt::constants::TILE_WIDTH))
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                        }
                        if (memory_config_o.is_sharded())
                        {
                            if (memory_config_o.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            uint32_t M =
                                (program_config.fuse_batch ? Volume(input_shape_a)/ input_shape_a[-1]
                                                        : input_shape_a[-2]) /
                                tt::constants::TILE_HEIGHT;
                            uint32_t N = input_shape_b[-1] / tt::constants::TILE_WIDTH;
                            uint32_t per_core_M = program_config.per_core_M;
                            uint32_t per_core_N = program_config.per_core_N;

                            if (N != per_core_N)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (program_config.out_subblock_w != per_core_N && program_config.out_subblock_h != 1)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                        }
                    }
                    if (memory_config_b.memory_layout != TensorMemoryLayout::INTERLEAVED)
                    {
                        return MatmulOpTypes::NotSupported;
                    }
                    if ((input_shape_a[-1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w != 0)
                    {
                        return MatmulOpTypes::NotSupported;
                    }
                    if (program_config.per_core_M % program_config.out_subblock_h != 0)
                    {
                        return MatmulOpTypes::NotSupported;
                    }
                    if (program_config.per_core_N % program_config.out_subblock_w != 0)
                    {
                        return MatmulOpTypes::NotSupported;
                    }
                }
                else if constexpr (std::is_same_v<T, ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                    if (memory_config_a.is_sharded())
                    {
                        auto tensor_a_memory_layout = memory_config_a.memory_layout;
                        uint32_t M = Volume(input_shape_a) / input_shape_a[-1] / tt::constants::TILE_HEIGHT;
                        uint32_t K = input_shape_a[-1] / tt::constants::TILE_WIDTH;
                        uint32_t N = input_shape_b[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = memory_config_a.shard_spec.value().shape;

                        if (tensor_a_memory_layout != TensorMemoryLayout::BLOCK_SHARDED && tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED)
                        {
                            return MatmulOpTypes::NotSupported;
                        }

                        if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                            if (program_config.transpose_mcast)
                            {
                                if (memory_config_a.shard_spec.value().orientation != ShardOrientation::COL_MAJOR)
                                {
                                    return MatmulOpTypes::NotSupported;
                                }
                            } else
                            {
                                if (memory_config_a.shard_spec.value().orientation != ShardOrientation::ROW_MAJOR)
                                {
                                    return MatmulOpTypes::NotSupported;
                                }
                            }
                            if (memory_config_o.is_sharded())
                            {
                                if (memory_config_a.buffer_type != memory_config_o.buffer_type)
                                {
                                    return MatmulOpTypes::NotSupported;
                                }
                                if (memory_config_a.memory_layout != memory_config_o.memory_layout)
                                {
                                    return MatmulOpTypes::NotSupported;
                                }
                            }

                        } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                            if (program_config.transpose_mcast)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (K != program_config.in0_block_w)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (program_config.in0_block_w != (shard_shape[1] / tt::constants::TILE_WIDTH))
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                            if (
                                memory_config_a.shard_spec.value().grid.bounding_box().start_coord.x !=
                                memory_config_a.shard_spec.value().grid.bounding_box().end_coord.x)
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                        }

                        if (per_core_M != (shard_shape[0] / tt::constants::TILE_HEIGHT))
                        {
                            return MatmulOpTypes::NotSupported;
                        }
                        if ((shard_shape[1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w != 0)
                        {
                            return MatmulOpTypes::NotSupported;
                        }
                    }

                    if (memory_config_b.is_sharded())
                    {
                        if (program_config.transpose_mcast)
                        {
                            return MatmulOpTypes::NotSupported;
                        }
                        auto tensor_b_memory_layout = memory_config_b.memory_layout;
                        if (tensor_b_memory_layout != TensorMemoryLayout::WIDTH_SHARDED)
                        {
                            return MatmulOpTypes::NotSupported;
                        }
                        if (memory_config_b.buffer_type != tt::tt_metal::BufferType::DRAM) {
                            if (program_config.per_core_N != (memory_config_b.shard_spec.value().shape[1] / tt::constants::TILE_WIDTH))
                            {
                                return MatmulOpTypes::NotSupported;
                            }
                        }
                        if (memory_config_b.shard_spec.value().grid.bounding_box().start_coord.x != memory_config_b.shard_spec.value().grid.bounding_box().end_coord.x)
                        {
                            return MatmulOpTypes::NotSupported;
                        }
                    }

                    if (memory_config_o.is_sharded()) {
                        if (memory_config_o.memory_layout != TensorMemoryLayout::BLOCK_SHARDED)
                        {
                            return MatmulOpTypes::NotSupported;
                        }
                        uint32_t M = Volume(input_shape_a) / input_shape_a[-1] / tt::constants::TILE_HEIGHT;
                        uint32_t N = input_shape_b[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        if (program_config.out_subblock_w != per_core_N && program_config.out_subblock_h != 1)
                        {
                            return MatmulOpTypes::NotSupported;
                        }
                    }
                    if ((input_shape_a[-1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w != 0)
                    {
                        return MatmulOpTypes::NotSupported;
                    }
                    if (program_config.per_core_M % program_config.out_subblock_h != 0)
                    {
                        return MatmulOpTypes::NotSupported;
                    }
                    if (program_config.per_core_N % program_config.out_subblock_w != 0)
                    {
                        return MatmulOpTypes::NotSupported;
                    }
                }
                else
                {
                    std::cout << "Currently not supported" << std::endl;
                    return MatmulOpTypes::NotSupported;
                }
                return MatmulOpTypes::NotSupported;
            },
            program_config);
        return MatmulOpTypes::NotSupported;
    }
};
