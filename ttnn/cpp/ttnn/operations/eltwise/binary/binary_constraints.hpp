#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
using EltwiseOpConstraint = std::tuple<
    // a
    tt::tt_metal::DataType,
    tt::tt_metal::Layout,
    tt::tt_metal::StorageType,
    // b
    tt::tt_metal::DataType,
    tt::tt_metal::Layout,
    tt::tt_metal::StorageType,
    // output
    tt::tt_metal::DataType,
    tt::tt_metal::Layout,
    tt::tt_metal::StorageType
>;

class EltwiseOpConstraintsBuilder {
    protected:
        const ttnn::Shape shape_a;
        const tt::tt_metal::MemoryConfig memory_config_a;
        const ttnn::Shape shape_b;
        const tt::tt_metal::MemoryConfig memory_config_b;
        const tt::tt_metal::MemoryConfig memory_config_o;

        std::optional<tt::tt_metal::DataType> data_type_a;  // required
        std::optional<tt::tt_metal::Layout> tile_layout_a;
        std::optional<tt::tt_metal::StorageType> storage_type_a;

        std::optional<tt::tt_metal::DataType> data_type_b;  // required
        std::optional<tt::tt_metal::Layout> tile_layout_b;
        std::optional<tt::tt_metal::StorageType> storage_type_b;

        std::optional<tt::tt_metal::DataType> data_type_o;  // required
        std::optional<tt::tt_metal::Layout> tile_layout_o;
        std::optional<tt::tt_metal::StorageType> storage_type_o;

        EltwiseOpConstraintsBuilder(const ttnn::Shape& _shape_a,
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
        bool is_valid_external_constraint(const EltwiseOpConstraint& constraint) const {
            if (data_type_a.has_value() && std::get<0>(constraint) != data_type_a.value()) {
                return false;
            }
            if (tile_layout_a.has_value() && std::get<1>(constraint) != tile_layout_a.value()) {
                return false;
            }
            if (storage_type_a.has_value() && std::get<2>(constraint) != storage_type_a.value()) {
                return false;
            }
            if (data_type_b.has_value() && std::get<3>(constraint) != data_type_b.value()) {
                return false;
            }
            if (tile_layout_b.has_value() && std::get<4>(constraint) != tile_layout_b.value()) {
                return false;
            }
            if (storage_type_b.has_value() && std::get<5>(constraint) != storage_type_b.value()) {
                return false;
            }
            if (data_type_o.has_value() && std::get<6>(constraint) != data_type_o.value()) {
                return false;
            }
            if (tile_layout_o.has_value() && std::get<7>(constraint) != tile_layout_o.value()) {
                return false;
            }
            if (storage_type_o.has_value() && std::get<8>(constraint) != storage_type_o.value()) {
                return false;
            }
            return true;
        }

        // op specific constraints
        virtual bool is_valid_op_constraint(const EltwiseOpConstraint& constraint) const = 0;

    public:
        virtual ~EltwiseOpConstraintsBuilder() = default;

        virtual std::string get_op_name() const = 0;

        std::vector<EltwiseOpConstraint> build_constraints()
        {
            if (can_build_constraints() == false)
            {
                throw std::runtime_error("Cannot build constraints, missing required parameters");
            }

            // data types are required
            static constexpr std::array<Layout, 2> tile_layouts = {Layout::ROW_MAJOR, Layout::TILE};
            // Only two for now. TODO: add other storage types.
            static constexpr std::array<StorageType, 2> storage_types = {StorageType::OWNED, StorageType::DEVICE};
            std::vector<EltwiseOpConstraint> constraints;

            // Currently we are only looking at softmax for one input.
            for (const auto& tile_layout_a : tile_layouts)
            {
                for (const auto& storage_type_a : storage_types)
                {
                    for (const auto& tile_layout_b : tile_layouts)
                    {
                        for (const auto& storage_type_b : storage_types)
                        {
                            for (const auto& tile_layout_o : tile_layouts)
                            {
                                for (const auto& storage_type_o : storage_types)
                                {
                                    const auto constraint = std::make_tuple(
                                        data_type_a.value(),
                                        tile_layout_a,
                                        storage_type_a,
                                        data_type_b.value(),
                                        tile_layout_b,
                                        storage_type_b,
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
                }
            }

            return std::move(constraints);
        }

        // Setters for parameter a
        EltwiseOpConstraintsBuilder& setDataTypeA(tt::tt_metal::DataType dataType) {
            data_type_a = dataType;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTileLayoutA(tt::tt_metal::Layout tileLayout) {
            tile_layout_a = tileLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setStorageTypeA(tt::tt_metal::StorageType storageType)
        {
            storage_type_a = storageType;
            return *this;
        }

        // Setters for parameter b
        EltwiseOpConstraintsBuilder& setDataTypeB(tt::tt_metal::DataType dataType) {
            data_type_b = dataType;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTileLayoutB(tt::tt_metal::Layout tileLayout) {
            tile_layout_b = tileLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setStorageTypeB(tt::tt_metal::StorageType storageType)
        {
            storage_type_b = storageType;
            return *this;
        }

        // Setters for parameter output
        EltwiseOpConstraintsBuilder& setDataTypeO(tt::tt_metal::DataType dataType) {
            data_type_o = dataType;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTileLayoutO(tt::tt_metal::Layout tileLayout) {
            tile_layout_o = tileLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setStorageTypeO(tt::tt_metal::StorageType storageType)
        {
            storage_type_o = storageType;
            return *this;
        }

};

enum class EltwiseOpTypes
{
    ElementWiseMultiCore,
    BroadcastWidthMultiCore,
    BroadcastHeightMultiCore,
    BroadcastHeightAndWidthMultiCore,
    BroadcastHeightMultiCoreSharded,
    BroadcastHeightMultiCoreShardedOptimized,
    NotSupported
};

class ElementWiseMultiCoreConstraintsBuilder : public EltwiseOpConstraintsBuilder
{
public:
    // ElementWiseMultiCoreConstraintsBuilder() = default;
    ElementWiseMultiCoreConstraintsBuilder(const ttnn::Shape& _shape_a,
                                           const tt::tt_metal::MemoryConfig& _memory_config_a,
                                           const ttnn::Shape& _shape_b,
                                           const tt::tt_metal::MemoryConfig& _memory_config_b,
                                           const tt::tt_metal::MemoryConfig& _memory_config_o)
    : EltwiseOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_b, _memory_config_b, _memory_config_o)
    {
        std::cout << "ElementWiseMultiCoreConstraintsBuilder" << std::endl;
    }
    virtual ~ElementWiseMultiCoreConstraintsBuilder() = default;

    virtual std::string get_op_name() const override {
        return "ElementWiseMultiCore";
    }

    static bool check_input_parameters(const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b)
    {
        auto height_a = input_shape_a[-2];
        auto width_a = input_shape_a[-1];

        auto height_b = input_shape_b[-2];
        auto width_b = input_shape_b[-1];

        if (height_a == height_b and width_a == width_b) {
            return true;
        }
        return false;
    }
protected:
    virtual bool is_valid_op_constraint(const EltwiseOpConstraint& constraint) const override {
        const tt::tt_metal::Layout c_tile_layout_a = std::get<1>(constraint);
        const tt::tt_metal::Layout c_tile_layout_b = std::get<4>(constraint);

        // made-up constraint - tiles only
        if (c_tile_layout_a != tt::tt_metal::Layout::TILE)
        {
            return false;
        }
        if (c_tile_layout_b != tt::tt_metal::Layout::TILE)
        {
            return false;
        }

        return true;
    }
};

class BroadcastWidthMultiCoreConstraintsBuilder : public EltwiseOpConstraintsBuilder
{
public:
    // BroadcastWidthMultiCoreConstraintsBuilder() = default;
    BroadcastWidthMultiCoreConstraintsBuilder(const ttnn::Shape& _shape_a,
                                              const tt::tt_metal::MemoryConfig& _memory_config_a,
                                              const ttnn::Shape& _shape_b,
                                              const tt::tt_metal::MemoryConfig& _memory_config_b,
                                              const tt::tt_metal::MemoryConfig& _memory_config_o)
    : EltwiseOpConstraintsBuilder(_shape_a, _memory_config_a, _shape_b, _memory_config_b, _memory_config_o)
    {
        std::cout << "BroadcastWidthMultiCoreConstraintsBuilder" << std::endl;
    }
    virtual ~BroadcastWidthMultiCoreConstraintsBuilder() = default;

    virtual std::string get_op_name() const override {
        return "BroadcastWidthMultiCoreConstraintsBuilder";
    }

    static bool check_input_parameters(const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b)
    {
        auto height_a = input_shape_a[-2];
        auto width_a = input_shape_a[-1];

        auto height_b = input_shape_b[-2];
        auto width_b = input_shape_b[-1];

        if (width_b == 1 && height_b > 1) {
            return true;
        }
        return false;
    }
protected:
    virtual bool is_valid_op_constraint(const EltwiseOpConstraint& constraint) const override {
        return true;
    }
};

class EltwiseOpConstraintsFactory
{
    public:
    EltwiseOpConstraintsFactory() = delete;
    static std::unique_ptr<EltwiseOpConstraintsBuilder> Make(const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b,
        const tt::tt_metal::MemoryConfig& memory_config_o)
    {
        auto eltwise_op_type = GetEltwiseOpType(input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
        switch (eltwise_op_type) {
            case EltwiseOpTypes::ElementWiseMultiCore:
                return std::make_unique<ElementWiseMultiCoreConstraintsBuilder>(input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
            case EltwiseOpTypes::BroadcastWidthMultiCore:
                return std::make_unique<BroadcastWidthMultiCoreConstraintsBuilder>(input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);
            case EltwiseOpTypes::BroadcastHeightMultiCore: // not implemented yet
            case EltwiseOpTypes::BroadcastHeightAndWidthMultiCore: // not implemented yet
            case EltwiseOpTypes::BroadcastHeightMultiCoreSharded: // not implemented yet
            case EltwiseOpTypes::BroadcastHeightMultiCoreShardedOptimized: // not implemented yet
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

    static EltwiseOpTypes GetEltwiseOpType(const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b,
        const tt::tt_metal::MemoryConfig& memory_config_o)
    {
        // void BinaryDeviceOperation::validate_on_program_cache_hit(
        auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
        auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
        auto height_a = input_shape_a[-2];
        auto width_a = input_shape_a[-1];

        auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
        auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
        auto height_b = input_shape_b[-2];
        auto width_b = input_shape_b[-1];

        // Input shape b must be the same as or broadcastable to input shape a
        if (batch_size_0_a != batch_size_0_b) {
            if (! (batch_size_0_a > batch_size_0_b and batch_size_0_b == 1))
            {
                // "ttnn::operations::binary::BinaryDeviceOperation: batch size mismatch");
                return EltwiseOpTypes::NotSupported;
            }
        }
        if (batch_size_1_a != batch_size_1_b) {
            if (! (batch_size_1_a > batch_size_1_b and batch_size_1_b == 1))
            {
                // "ttnn::operations::binary::BinaryDeviceOperation: batch size mismatch");
                return EltwiseOpTypes::NotSupported;
            }
        }
        if (height_a != height_b) {
            if (! (height_a > height_b and height_b == 1))
            {
                // "ttnn::operations::binary::BinaryDeviceOperation: height mismatch");
                return EltwiseOpTypes::NotSupported;
            }
        }
        if (width_a != width_b) {
            if (! (width_a > width_b and width_b == 1))
            {
                // "ttnn::operations::binary::BinaryDeviceOperation: width mismatch");
                return EltwiseOpTypes::NotSupported;
            }
        }
        if (memory_config_a.is_sharded()) {
            if (memory_config_a.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED)
            {
                // If we aren't height sharded, we require all sharding schemes to match until we add blocked
                // reader/writers for width and block sharding
                if (!(memory_config_b.is_sharded()))
                {
                    return EltwiseOpTypes::NotSupported;
                }
                if (memory_config_a.shard_spec.value().grid.ranges().size() != 1)
                {
                    return EltwiseOpTypes::NotSupported;
                }
            }
            if (memory_config_b.is_sharded())
            {
                if (memory_config_a.memory_layout != memory_config_b.memory_layout)
                {
                    return EltwiseOpTypes::NotSupported;
                }
                if (memory_config_a.shard_spec.value() == memory_config_b.shard_spec.value())
                {
                    return EltwiseOpTypes::NotSupported;
                }
            }
            if (memory_config_o.is_sharded())
            {
                if (memory_config_a.memory_layout != memory_config_o.memory_layout)
                {
                    return EltwiseOpTypes::NotSupported;
                }
            } else {
                if (memory_config_o.memory_layout != TensorMemoryLayout::INTERLEAVED)
                {
                    return EltwiseOpTypes::NotSupported;
                }
            }
        }
        else if (memory_config_b.is_sharded())
        {
            if (memory_config_b.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED)
            {
                return EltwiseOpTypes::NotSupported;
            }
            if (memory_config_a.memory_layout != TensorMemoryLayout::INTERLEAVED)
            {
                return EltwiseOpTypes::NotSupported;
            }
            if (memory_config_o.is_sharded()) {
                if (memory_config_b.memory_layout != memory_config_o.memory_layout)
                {
                    return EltwiseOpTypes::NotSupported;
                }
            } else
            {
                if (memory_config_b.memory_layout != TensorMemoryLayout::INTERLEAVED)
                {
                    return EltwiseOpTypes::NotSupported;
                }
            }
        }
        else
        {
            if (memory_config_a.memory_layout != TensorMemoryLayout::INTERLEAVED)
            {
                return EltwiseOpTypes::NotSupported;
            }
            if (memory_config_b.memory_layout != TensorMemoryLayout::INTERLEAVED)
            {
                return EltwiseOpTypes::NotSupported;
            }
            if (memory_config_o.is_sharded())
            {
                if (memory_config_o.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED)
                {
                    return EltwiseOpTypes::NotSupported;
                }
                // Check if we need this.
                /*uint32_t num_blocks = Volume(input_shape_a) / input_shape_a[-1] / tt::constants::TILE_HEIGHT;
                auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
                uint32_t num_cores = core_grid.x * core_grid.y;
                if (num_blocks < num_cores or num_blocks % num_cores != 0)
                {
                    return EltwiseOpTypes::NotSupported;
                }*/
            }
            else {
                if (memory_config_o.memory_layout != TensorMemoryLayout::INTERLEAVED)
                {
                    return EltwiseOpTypes::NotSupported;
                }
            }
        }

        if (ElementWiseMultiCoreConstraintsBuilder::check_input_parameters(input_shape_a, memory_config_a, input_shape_b, memory_config_b)) {
            return EltwiseOpTypes::ElementWiseMultiCore;
        } else if (BroadcastWidthMultiCoreConstraintsBuilder::check_input_parameters(input_shape_a, memory_config_a, input_shape_b, memory_config_b)) {
            return EltwiseOpTypes::BroadcastWidthMultiCore;
        }
        std::cout << "EltwiseOpTypes::NotSupported" << std::endl;
        // todo other op flavors

        return EltwiseOpTypes::NotSupported;
    }
};
