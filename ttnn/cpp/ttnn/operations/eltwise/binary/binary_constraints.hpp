#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace detail { // constexpr combinations
template <typename Array, std::size_t... I>
constexpr auto array_to_tuple_impl(const Array& a, std::index_sequence<I...>)
{
    return std::make_tuple(a[I]...);
}

template <typename T, std::size_t N, typename Indx = std::make_index_sequence<N>>
constexpr auto array_to_tuple(const std::array<T, N>& a)
{
    return array_to_tuple_impl(a, Indx{});
}

template<typename Tuple, typename NewElement>
constexpr auto extendTupleDirect(const Tuple& t, NewElement&& newElem) {
    return std::tuple_cat(t, std::make_tuple(std::forward<NewElement>(newElem)));
}

template <typename Array1, typename Array2, std::size_t... I, std::size_t... J, std::size_t... E>
constexpr auto combine_array_x_array_impl(const Array1& a1, const Array2& a2, std::index_sequence<I...>, std::index_sequence<J...>, std::index_sequence<E...>)
// template <typename Array1, typename Array2, std::size_t... I, std::size_t... J>
// constexpr auto arrays_to_tuple_cartesian_impl(const Array1& a1, const Array2& a2, std::index_sequence<I...>, std::index_sequence<J...>)
{
    // there is probably a better way to achieve this
    size_t i = 0;
    size_t j = 0;
    return std::array<std::tuple<typename Array1::value_type, typename Array2::value_type>, sizeof...(I) * sizeof...(J)> {
        [&i, &j, a1, a2](auto...) {
            if (i * j >= sizeof...(I) * sizeof...(J)) {
                return std::tuple<typename Array1::value_type, typename Array2::value_type>{};
            }

            auto value = std::make_tuple(a1[i], a2[j]);
            if (++j >= sizeof...(J)) {
                j = 0;
                ++i;
            }
            return value;
        }(std::integral_constant<std::size_t, E>{})...};
}

template <typename T, std::size_t N, typename U, std::size_t M, typename Indx1 = std::make_index_sequence<N>, typename Indx2 = std::make_index_sequence<M>>
constexpr auto combine_array_x_array(const std::array<T, N>& a1, const std::array<U, M>& a2)
{
    return combine_array_x_array_impl(a1, a2, Indx1{}, Indx2{}, std::make_index_sequence<N * M>{});
}

template <typename... TupleArgs, typename U, std::size_t... I, std::size_t... J, std::size_t... E>
constexpr auto combine_tuple_x_array_impl(
    const std::array<std::tuple<TupleArgs...>, sizeof...(I)> & a1,
    const std::array<U, sizeof...(J)>& a2,
    std::index_sequence<I...>,
    std::index_sequence<J...>,
    std::index_sequence<E...>)
{
    // there is probably a better way to achieve this
    size_t i = 0;
    size_t j = 0;
    return std::array<std::tuple<TupleArgs..., U>, sizeof...(I) * sizeof...(J)> {
        [&i, &j, a1, a2](auto...) {
            if (i * j >= sizeof...(I) * sizeof...(J)) {
                return std::tuple<TupleArgs..., U>{};
            }
            auto value = extendTupleDirect(a1[i], a2[j]);
            if (++j >= sizeof...(J)) {
                j = 0;
                ++i;
            }
            return value;
        }(std::integral_constant<std::size_t, E>{})...};
}

template <typename... TupleArgs, std::size_t N, typename U, std::size_t M, typename Indx1 = std::make_index_sequence<N>, typename Indx2 = std::make_index_sequence<M>>
constexpr auto combine_tuple_x_array(const std::array<std::tuple<TupleArgs...>, N>& a1, const std::array<U, M>& a2)
{
    return combine_tuple_x_array_impl(a1, a2, Indx1{}, Indx2{}, std::make_index_sequence<N * M>{});
}
}

// input shapes and shard shapes are runtime thing, we cannot return arbitrary count of shapes

using EltwiseOpConstraint = std::tuple<
    // a
    tt::tt_metal::DataType,
    tt::tt_metal::BufferType,
    bool,
    tt::tt_metal::Layout,
    tt::tt_metal::TensorMemoryLayout,
    tt::tt_metal::ShardOrientation,
    // b
    tt::tt_metal::DataType,
    tt::tt_metal::BufferType,
    bool,
    tt::tt_metal::Layout,
    tt::tt_metal::TensorMemoryLayout,
    tt::tt_metal::ShardOrientation,
    // output
    tt::tt_metal::DataType,
    tt::tt_metal::BufferType,
    bool,
    tt::tt_metal::Layout,
    tt::tt_metal::TensorMemoryLayout,
    tt::tt_metal::ShardOrientation
>;

class EltwiseOpConstraintsBuilder {
    protected:
        std::optional<tt::tt_metal::DataType> data_type_a;  // required
        std::optional<tt::tt_metal::BufferType> buffer_type_a; // required
        std::optional<bool> is_sharded_a; // required
        std::optional<tt::tt_metal::Layout> tile_layout_a;
        std::optional<tt::tt_metal::TensorMemoryLayout> tensor_memory_layout_a;
        std::optional<tt::tt_metal::ShardOrientation> shard_orientation_a;

        std::optional<tt::tt_metal::DataType> data_type_b;  // required
        std::optional<tt::tt_metal::BufferType> buffer_type_b; // required
        std::optional<bool> is_sharded_b; // required
        std::optional<tt::tt_metal::Layout> tile_layout_b;
        std::optional<tt::tt_metal::TensorMemoryLayout> tensor_memory_layout_b;
        std::optional<tt::tt_metal::ShardOrientation> shard_orientation_b;

        std::optional<tt::tt_metal::DataType> data_type_o;  // required
        std::optional<tt::tt_metal::BufferType> buffer_type_o; // required
        std::optional<bool> is_sharded_o; // required
        std::optional<tt::tt_metal::Layout> tile_layout_o;
        std::optional<tt::tt_metal::TensorMemoryLayout> tensor_memory_layout_o;
        std::optional<tt::tt_metal::ShardOrientation> shard_orientation_o;

        // check if required parameters are set
        bool can_build_constraints() const
        {
            return data_type_a.has_value() && buffer_type_a.has_value() && is_sharded_a.has_value() &&
                data_type_b.has_value() && buffer_type_b.has_value() && is_sharded_b.has_value() &&
                data_type_o.has_value() && buffer_type_o.has_value() && is_sharded_o.has_value();
        }

        // check if it is possible to build constraints with all set parameters
        bool is_valid_external_constraint(const EltwiseOpConstraint& constraint) const {
            if (data_type_a.has_value() && std::get<0>(constraint) != data_type_a.value()) {
                return false;
            }
            if (buffer_type_a.has_value() && std::get<1>(constraint) != buffer_type_a.value()) {
                return false;
            }
            if (is_sharded_a.has_value() && std::get<2>(constraint) != is_sharded_a.value()) {
                return false;
            }
            if (tile_layout_a.has_value() && std::get<3>(constraint) != tile_layout_a.value()) {
                return false;
            }
            if (tensor_memory_layout_a.has_value() && std::get<4>(constraint) != tensor_memory_layout_a.value()) {
                return false;
            }
            if (shard_orientation_a.has_value() && std::get<5>(constraint) != shard_orientation_a.value()) {
                return false;
            }
            if (data_type_b.has_value() && std::get<6>(constraint) != data_type_b.value()) {
                return false;
            }
            if (buffer_type_b.has_value() && std::get<7>(constraint) != buffer_type_b.value()) {
                return false;
            }
            if (is_sharded_b.has_value() && std::get<8>(constraint) != is_sharded_b.value()) {
                return false;
            }
            if (tile_layout_b.has_value() && std::get<9>(constraint) != tile_layout_b.value()) {
                return false;
            }
            if (tensor_memory_layout_b.has_value() && std::get<10>(constraint) != tensor_memory_layout_b.value()) {
                return false;
            }
            if (shard_orientation_b.has_value() && std::get<11>(constraint) != shard_orientation_b.value()) {
                return false;
            }
            if (data_type_o.has_value() && std::get<12>(constraint) != data_type_o.value()) {
                return false;
            }
            if (buffer_type_o.has_value() && std::get<13>(constraint) != buffer_type_o.value()) {
                return false;
            }
            if (is_sharded_o.has_value() && std::get<14>(constraint) != is_sharded_o.value()) {
                return false;
            }
            if (tile_layout_o.has_value() && std::get<15>(constraint) != tile_layout_o.value()) {
                return false;
            }
            if (tensor_memory_layout_o.has_value() && std::get<16>(constraint) != tensor_memory_layout_o.value()) {
                return false;
            }
            if (shard_orientation_o.has_value() && std::get<17>(constraint) != shard_orientation_o.value()) {
                return false;
            }
            return true;
        }

        // op specific constraints
        virtual bool is_valid_op_constraint(const EltwiseOpConstraint& constraint) const = 0;

    public:
        EltwiseOpConstraintsBuilder() = default;
        virtual ~EltwiseOpConstraintsBuilder() = default;

        virtual std::string get_op_name() const = 0;

        std::vector<EltwiseOpConstraint> build_constraints()
        {
            if (can_build_constraints() == false)
            {
                throw std::runtime_error("Cannot build constraints, missing required parameters");
            }

            // reducing search space
            // data types are required
            // buffer types are required
            // is sharded is required
            std::vector<TensorMemoryLayout> sweep_tensor_memory_layout_a;
            if (is_sharded_a.value()) {
                sweep_tensor_memory_layout_a = {TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::BLOCK_SHARDED};
            }
            else
            {
                sweep_tensor_memory_layout_a = {TensorMemoryLayout::INTERLEAVED};
            }
            std::vector<TensorMemoryLayout> sweep_tensor_memory_layout_b;
            if (is_sharded_b.value()) {
                sweep_tensor_memory_layout_b = {TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::BLOCK_SHARDED};
            }
            else
            {
                sweep_tensor_memory_layout_b = {TensorMemoryLayout::INTERLEAVED};
            }
            std::vector<TensorMemoryLayout> sweep_tensor_memory_layout_o;
            if (is_sharded_o.value()) {
                sweep_tensor_memory_layout_o = {TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::BLOCK_SHARDED};
            }
            else
            {
                sweep_tensor_memory_layout_o = {TensorMemoryLayout::INTERLEAVED};
            }
            static constexpr std::array<ShardOrientation, 2> shard_shapes = {ShardOrientation::ROW_MAJOR, ShardOrientation::COL_MAJOR};
            static constexpr std::array<Layout, 2> tile_layouts = {Layout::ROW_MAJOR, Layout::TILE};

            std::vector<EltwiseOpConstraint> constraints;
            for (const auto& tml_a : sweep_tensor_memory_layout_a)
            {
                for (const auto& tml_b : sweep_tensor_memory_layout_b)
                {
                    for (const auto& tml_o : sweep_tensor_memory_layout_o)
                    {
                        for (const auto& shard_shape_a : shard_shapes)
                        {
                            for (const auto& shard_shape_b : shard_shapes)
                            {
                                for (const auto& shard_shape_o : shard_shapes)
                                {
                                    for (const auto& tile_layout_a : tile_layouts)
                                    {
                                        for (const auto& tile_layout_b : tile_layouts)
                                        {
                                            for (const auto& tile_layout_o : tile_layouts)
                                            {
                                                const auto constraint = std::make_tuple(
                                                    data_type_a.value(),
                                                    buffer_type_a.value(),
                                                    is_sharded_a.value(),
                                                    tile_layout_a,
                                                    tml_a,
                                                    shard_shape_a,
                                                    data_type_b.value(),
                                                    buffer_type_b.value(),
                                                    is_sharded_b.value(),
                                                    tile_layout_b,
                                                    tml_b,
                                                    shard_shape_b,
                                                    data_type_o.value(),
                                                    buffer_type_o.value(),
                                                    is_sharded_o.value(),
                                                    tile_layout_o,
                                                    tml_o,
                                                    shard_shape_o
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

        EltwiseOpConstraintsBuilder& setBufferTypeA(tt::tt_metal::BufferType bufferType) {
            buffer_type_a = bufferType;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setIsShardedA(bool isSharded) {
            is_sharded_a = isSharded;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTileLayoutA(tt::tt_metal::Layout tileLayout) {
            tile_layout_a = tileLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTensorMemoryLayoutA(tt::tt_metal::TensorMemoryLayout tensorMemoryLayout) {
            tensor_memory_layout_a = tensorMemoryLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setShardOrientationA(tt::tt_metal::ShardOrientation shardOrientation) {
            shard_orientation_a = shardOrientation;
            return *this;
        }

        // Setters for parameter b
        EltwiseOpConstraintsBuilder& setDataTypeB(tt::tt_metal::DataType dataType) {
            data_type_b = dataType;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setBufferTypeB(tt::tt_metal::BufferType bufferType) {
            buffer_type_b = bufferType;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setIsShardedB(bool isSharded) {
            is_sharded_b = isSharded;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTileLayoutB(tt::tt_metal::Layout tileLayout) {
            tile_layout_b = tileLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTensorMemoryLayoutB(tt::tt_metal::TensorMemoryLayout tensorMemoryLayout) {
            tensor_memory_layout_b = tensorMemoryLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setShardOrientationB(tt::tt_metal::ShardOrientation shardOrientation) {
            shard_orientation_b = shardOrientation;
            return *this;
        }

        // Setters for parameter output
        EltwiseOpConstraintsBuilder& setDataTypeO(tt::tt_metal::DataType dataType) {
            data_type_o = dataType;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setBufferTypeO(tt::tt_metal::BufferType bufferType) {
            buffer_type_o = bufferType;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setIsShardedO(bool isSharded) {
            is_sharded_o = isSharded;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTileLayoutO(tt::tt_metal::Layout tileLayout) {
            tile_layout_o = tileLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setTensorMemoryLayoutO(tt::tt_metal::TensorMemoryLayout tensorMemoryLayout) {
            tensor_memory_layout_o = tensorMemoryLayout;
            return *this;
        }

        EltwiseOpConstraintsBuilder& setShardOrientationO(tt::tt_metal::ShardOrientation shardOrientation) {
            shard_orientation_o = shardOrientation;
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
    ElementWiseMultiCoreConstraintsBuilder() {
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
        const tt::tt_metal::Layout c_tile_layout_a = std::get<3>(constraint);
        const tt::tt_metal::Layout c_tile_layout_b = std::get<9>(constraint);
        const tt::tt_metal::Layout c_tile_layout_o = std::get<15>(constraint);

        // made-up constraint - tiles only
        if (c_tile_layout_a != tt::tt_metal::Layout::TILE)
        {
            return false;
        }
        if (c_tile_layout_b != tt::tt_metal::Layout::TILE)
        {
            return false;
        }
        if (c_tile_layout_o != tt::tt_metal::Layout::TILE)
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
    BroadcastWidthMultiCoreConstraintsBuilder() {
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
        const bool c_is_sharded_a = std::get<2>(constraint);
        const bool c_is_sharded_b = std::get<8>(constraint);
        const bool c_is_sharded_o = std::get<14>(constraint);

        const tt::tt_metal::TensorMemoryLayout c_tensor_memory_layout_a = std::get<4>(constraint);
        const tt::tt_metal::TensorMemoryLayout c_tensor_memory_layout_b = std::get<10>(constraint);
        const tt::tt_metal::TensorMemoryLayout c_tensor_memory_layout_c = std::get<16>(constraint);

        // BroadcastWidthMultiCoreConstraintsBuilder doesn't support sharding
        if (c_is_sharded_a)
        {
            return false;
        }
        if (c_is_sharded_b)
        {
            return false;
        }
        if (c_is_sharded_o)
        {
            return false;
        }
        if (c_tensor_memory_layout_a != tt::tt_metal::TensorMemoryLayout::INTERLEAVED)
        {
            return false;
        }
        if (c_tensor_memory_layout_b != tt::tt_metal::TensorMemoryLayout::INTERLEAVED)
        {
            return false;
        }
        if (c_tensor_memory_layout_c != tt::tt_metal::TensorMemoryLayout::INTERLEAVED)
        {
            return false;
        }

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
        tt::tt_metal::MemoryConfig& memory_config_b)
    {
        auto eltwise_op_type = GetEltwiseOpType(input_shape_a, memory_config_a, input_shape_b, memory_config_b);
        switch (eltwise_op_type) {
            case EltwiseOpTypes::ElementWiseMultiCore:
                return std::make_unique<ElementWiseMultiCoreConstraintsBuilder>();
            case EltwiseOpTypes::BroadcastWidthMultiCore:
                return std::make_unique<BroadcastWidthMultiCoreConstraintsBuilder>();
            case EltwiseOpTypes::BroadcastHeightMultiCore: // not implemented yet
            case EltwiseOpTypes::BroadcastHeightAndWidthMultiCore: // not implemented yet
            case EltwiseOpTypes::BroadcastHeightMultiCoreSharded: // not implemented yet
            case EltwiseOpTypes::BroadcastHeightMultiCoreShardedOptimized: // not implemented yet
            default:
                return nullptr;
        }
    };

    static EltwiseOpTypes GetEltwiseOpType(const ttnn::Shape& input_shape_a,
        const tt::tt_metal::MemoryConfig& memory_config_a,
        const ttnn::Shape& input_shape_b,
        const tt::tt_metal::MemoryConfig& memory_config_b)
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


// constexpr EltwiseOpConstraintsBuilder constraints_builder = EltwiseOpConstraintsBuilder();

// constexpr ttnn::Shape a = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32});

// constexpr tt::tt_metal::MemoryConfig memory_config_a = {
//     .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
//     .buffer_type = tt::tt_metal::BufferType::L1,
//     .shard_spec = tt::tt_metal::ShardSpec{
//         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
//         {160, 32},
//         ShardOrientation::COL_MAJOR
//     }
// };

// constexpr tt::tt_metal::ShardSpec ss = {
//         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
//         {160, 32},
//         ShardOrientation::COL_MAJOR
//     };
// ../ttnn/cpp/ttnn/operations/eltwise/binary/binary_constraints.hpp:20:35: error: constexpr variable cannot have non-literal type 'const tt::tt_metal::ShardSpec'
//    20 | constexpr tt::tt_metal::ShardSpec ss = {
//       |                                   ^
// ../tt_metal/impl/buffers/buffer.hpp:38:8: note: 'ShardSpec' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors
//    38 | struct ShardSpec {

// bool is_binary_op_valid(
//         const ttnn::Shape& input_shape_a,
//         const tt::tt_metal::MemoryConfig& memory_config_a,
//         const ttnn::Shape& input_shape_b,
//         tt::tt_metal::MemoryConfig& memory_config_b) {

//         auto height_a = input_shape_a[-2];
//         auto width_a = input_shape_a[-1];

//         auto height_b = input_shape_b[-2];
//         auto width_b = input_shape_b[-1];

//     if (height_a == height_b and width_a == width_b) {
//         std::cout << "ElementWiseMultiCore" << std::endl;
//         // return ElementWiseMultiCore{};
//         return true;
//     } else if (height_b == 1 or width_b == 1) {
//         if (height_b == 1 and width_b == 1) {
//             std::cout << "BroadcastHeightAndWidthMultiCore" << std::endl;
//             // return BroadcastHeightAndWidthMultiCore{};
//             // no additional constraints at
//             // BinaryDeviceOperation::ElementWiseMultiCore::cached_program_t BinaryDeviceOperation::ElementWiseMultiCore::create(
//             return true;
//         } else if (height_b == 1) {
//             if(memory_config_a.is_sharded()){
//                 if (input_shape_a.value[0] == input_shape_b.value[0]
//                         || input_shape_a.value[0] > 1
//                         and input_shape_b.value[0] == 1){
//                             std::cout << "BroadcastHeightMultiCoreShardedOptimized" << std::endl;

//                             //  "Output tensor should have same number of cores {} as input tensor {}",
//                             // TODO needs output

//                             //  "Input and output tile size should be same"
//                             // TODO needs output

//                             //  "Input tile size should be less than shard size"
//                             // TODO needs data format

//                             if (memory_config_a.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
//                             //     ncores_x = all_cores.ranges().begin()->end_coord.y + 1;
//                             //     Wt = shard_spec.shape[1] / TILE_WIDTH;
//                             //     Ht = shard_spec.shape[0] / TILE_HEIGHT;
//                             } else if (memory_config_a.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
//                                 uint32_t bN = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
//                                 if (memory_config_a.shard_spec.value().shape[0] % (bN * tt::constants::TILE_HEIGHT) != 0)
//                                 {
//                                     return false;
//                                 }
//                             //     Wt = shard_spec.shape[1] / TILE_WIDTH;
//                             //     Ht = shard_spec.shape[0] / TILE_HEIGHT;
//                             //     TT_ASSERT(
//                             //         (shard_spec.shape[0] % (bN * TILE_HEIGHT) == 0),
//                             //         "Shard height per batch must be divisible by TILE_HEIGHT {} {} {} ",
//                             //         shard_spec.shape[0],
//                             //         bN,
//                             //         TILE_HEIGHT);
//                             } else {
//                             //     TT_FATAL(false, "1 Unsupported memory layout");
//                                 return false;
//                             }

//                             if (memory_config_a.shard_spec.value().shape[0] % tt::constants::TILE_HEIGHT != 0)
//                             {
//                                 return false;
//                             }
//                             if (memory_config_a.shard_spec.value().shape[0] % tt::constants::TILE_WIDTH != 0)
//                             {
//                                 return false;
//                             }
//                             // TT_ASSERT(
//                                 // (shard_spec.shape[0] % TILE_HEIGHT == 0) && (shard_spec.shape[0] % TILE_WIDTH == 0),
//                                 // "Shard shapes must be multiple of TILE_HEIGHT ");

//                             uint32_t N = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
//                             uint32_t C = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
//                             uint32_t H = input_shape_a[-2];
//                             uint32_t W = input_shape_a[-1];
//                             uint32_t bN = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
//                             uint32_t bC = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
//                             uint32_t bH = input_shape_b[-2];
//                             uint32_t bW = input_shape_b[-1];
//                             uint32_t NC = N * C;
//                             uint32_t HW = H * W;
//                             if ((NC * H / tt::constants::TILE_HEIGHT) % bN != 0)
//                             {
//                                 return false;
//                             }
//                             //     TT_FATAL((NC * H / TILE_HEIGHT) % bN == 0, "N*C*H of input0 must be devisible by batch size of input1");

//                         // return BroadcastHeightMultiCoreShardedOptimized{};
//                         return true;
//                 } else {
//                     std::cout << "BroadcastHeightMultiCoreSharded" << std::endl;
//                         // return BroadcastHeightMultiCoreSharded{};
//                         return true;
//                 }
//             }
//             std::cout << "BroadcastHeightMultiCore" << std::endl;
//             // return BroadcastHeightMultiCore{};
//             return true;
//         } else if (width_b == 1) {
//             std::cout << "BroadcastWidthMultiCore" << std::endl;
//             // return BroadcastWidthMultiCore{};
//             // this is obivous only if you look at BroadcastWidthMultiCore::create
//             if (memory_config_a.is_sharded() || memory_config_b.is_sharded())
//             {
//                 return false;
//             }
//             return true;
//         }
//     }
//         return false;
//     };
