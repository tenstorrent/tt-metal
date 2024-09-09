#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
// input shapes and shard shapes are runtime thing, we cannot return arbitrary count of shapes

// Currently we are only looking at softmax for one input.
using SoftmaxOpConstraint = std::tuple<
    // a
    tt::tt_metal::DataType,
    tt::tt_metal::Layout,
    // output
    tt::tt_metal::DataType,
    tt::tt_metal::Layout
>;

class SoftmaxOpConstraintsBuilder {
    protected:
        const ttnn::Shape shape_a;
        const tt::tt_metal::MemoryConfig memory_config_a;
        const ttnn::Shape shape_o;
        const tt::tt_metal::MemoryConfig memory_config_o;

        std::optional<tt::tt_metal::DataType> data_type_a;  // required
        std::optional<tt::tt_metal::Layout> tile_layout_a;

        std::optional<tt::tt_metal::DataType> data_type_o; // required
        std::optional<tt::tt_metal::Layout> tile_layout_o;

        SoftmaxOpConstraintsBuilder(const ttnn::Shape& _shape_a,
                                    const tt::tt_metal::MemoryConfig& _memory_config_a,
                                    const ttnn::Shape& _shape_o,
                                    const tt::tt_metal::MemoryConfig& _memory_config_o) :
                                    shape_a(_shape_a), memory_config_a(_memory_config_a),
                                    shape_o(_shape_o), memory_config_o(_memory_config_o) {}

        // check if required parameters are set
        bool can_build_constraints() const
        {
            return (data_type_a.has_value() && data_type_o.has_value());
        }

        // check if it is possible to build constraints with all set parameters
        bool is_valid_external_constraint(const SoftmaxOpConstraint& constraint) const {
            if (data_type_a.has_value() && std::get<0>(constraint) != data_type_a.value()) {
                return false;
            }
            if (tile_layout_a.has_value() && std::get<1>(constraint) != tile_layout_a.value()) {
                return false;
            }
            if (data_type_o.has_value() && std::get<2>(constraint) != data_type_o.value()) {
                return false;
            }
            if (tile_layout_o.has_value() && std::get<3>(constraint) != tile_layout_o.value()) {
                return false;
            }
            return true;
        }

        // op specific constraints
        virtual bool is_valid_op_constraint(const SoftmaxOpConstraint& constraint) const = 0;

    public:
        virtual ~SoftmaxOpConstraintsBuilder() = default;

        virtual std::string get_op_name() const = 0;

        std::vector<SoftmaxOpConstraint> build_constraints()
        {
            if (can_build_constraints() == false)
            {
                throw std::runtime_error("Cannot build constraints, missing required parameters");
            }

            // reducing search space
            // data types are required
            static constexpr std::array<Layout, 2> tile_layouts = {Layout::ROW_MAJOR, Layout::TILE};

            std::vector<SoftmaxOpConstraint> constraints;

            // Currently we are only looking at softmax for one input.
            for (const auto& tile_layout_a : tile_layouts)
            {
                for (const auto& tile_layout_o : tile_layouts)
                {
                    const auto constraint = std::make_tuple(
                        data_type_a.value(),
                        tile_layout_a,
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
            return std::move(constraints);
        }

        // Setters for parameter a
        SoftmaxOpConstraintsBuilder& setDataTypeA(tt::tt_metal::DataType dataType) {
            data_type_a = dataType;
            return *this;
        }

        SoftmaxOpConstraintsBuilder& setTileLayoutA(tt::tt_metal::Layout tileLayout) {
            tile_layout_a = tileLayout;
            return *this;
        }

        // Setters for parameter output
        SoftmaxOpConstraintsBuilder& setDataTypeO(tt::tt_metal::DataType dataType) {
            data_type_o = dataType;
            return *this;
        }

        SoftmaxOpConstraintsBuilder& setTileLayoutO(tt::tt_metal::Layout tileLayout) {
            tile_layout_o = tileLayout;
            return *this;
        }
};

enum class SoftmaxOpTypes
{
    SoftmaxInPlace,
    ScaleMaskSoftmaxInPlace,
    ScaleCausalMaskHwDimsSoftmaxInPlace,
    Softmax,
    ScaleMaskSoftmax,
    NotSupported
};

class SoftmaxConstraintsBuilder : public SoftmaxOpConstraintsBuilder
{
public:
    // ElementWiseMultiCoreConstraintsBuilder() = default;
    SoftmaxConstraintsBuilder(const ttnn::Shape& _shape_a,
                              const tt::tt_metal::MemoryConfig& _memory_config_a,
                              const ttnn::Shape& _shape_o,
                              const tt::tt_metal::MemoryConfig& _memory_config_o)
                              : SoftmaxOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_o, _memory_config_o)
    {
        std::cout << "SoftmaxConstraintsBuilder" << std::endl;
    }
    virtual ~SoftmaxConstraintsBuilder() = default;

    virtual std::string get_op_name() const override {
        return "Softmax";
    }

protected:
    virtual bool is_valid_op_constraint(const SoftmaxOpConstraint& constraint) const override {
        const tt::tt_metal::DataType data_type_a = std::get<0>(constraint);
        const tt::tt_metal::Layout c_tile_layout_a = std::get<3>(constraint);
        // made-up constraint - tiles only
        if (c_tile_layout_a != tt::tt_metal::Layout::TILE)
        {
            return false;
        }
        if (!(data_type_a == tt::tt_metal::DataType::FLOAT32 || data_type_a == tt::tt_metal::DataType::BFLOAT16 || data_type_a == tt::tt_metal::DataType::BFLOAT8_B))
        {
            return false;
        }

        return true;
    }
};
class SoftmaxOpConstraintsFactory
{
    public:
    SoftmaxOpConstraintsFactory() = delete;
    static std::unique_ptr<SoftmaxOpConstraintsBuilder> Make(
        const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_o,
        const tt::tt_metal::MemoryConfig& memory_config_o,
        std::optional<const ttnn::Shape>& input_shape_b,
        std::optional<const tt::tt_metal::MemoryConfig>& memory_config_b)
    {
        auto Softmax_op_type = GetSoftmaxOpType(input_shape_a, memory_config_a, input_shape_b, memory_config_b);
        switch (Softmax_op_type) {
            case SoftmaxOpTypes::Softmax:
                return std::make_unique<SoftmaxConstraintsBuilder>(input_shape_a, memory_config_a, input_shape_o, memory_config_o);
            case SoftmaxOpTypes::SoftmaxInPlace:
            case SoftmaxOpTypes::ScaleMaskSoftmaxInPlace: // not implemented yet
            case SoftmaxOpTypes::ScaleCausalMaskHwDimsSoftmaxInPlace: // not implemented yet
            case SoftmaxOpTypes::ScaleMaskSoftmax: // not implemented yet
            default:
                return nullptr;
        }
    };

    static SoftmaxOpTypes GetSoftmaxOpType(const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        std::optional<const ttnn::Shape>& input_shape_b,
        std::optional<const tt::tt_metal::MemoryConfig>& memory_config_b)
    {

        if (input_shape_b.has_value() || memory_config_b.has_value())
        {
            std::cout << "SoftmaxOpTypes::NotSupported" << std::endl;
            return SoftmaxOpTypes::NotSupported;
        }
        return SoftmaxOpTypes::Softmax;
    }
};
