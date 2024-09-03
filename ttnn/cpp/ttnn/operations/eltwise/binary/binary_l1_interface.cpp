#include "binary_l1_interface.hpp"
#include <optional>
#include <tuple>
#include <memory>

#include "tt_metal/common/constants.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor_impl.hpp"


uint32_t get_num_of_cores(const std::optional<tt::tt_metal::ShardSpec>& shard_spec) {
    if (shard_spec.has_value()) {
        return shard_spec.value().grid.num_cores();
    }
    return (uint32_t)64; // Nebula_x1
};

ttnn::types::Shape get_larger_shape_by_volume(const ttnn::types::Shape& a, const ttnn::types::Shape& b) {
    if (a.volume() > b.volume()) {
        return a;
    }
    else {
        return b;
    }
};

EltwiseOpParams get_larger_eltwise_op_params_by_volume(const EltwiseOpParams& a, const EltwiseOpParams& b) {
    if (std::get<ttnn::types::Shape>(a).volume() > std::get<ttnn::types::Shape>(b).volume()) {
        return a;
    }
    else {
        return b;
    }
};

// these should go to a common tensor_l1_interface header or something
uint32_t calculate_circular_buffer_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config)
{
    auto total_size_bytes = shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
    auto page_size = tt::tt_metal::tensor_impl::get_page_size(data_type, layout, total_size_bytes, shape.value);
    auto num_pages = 2; /// default value for interleaved
    if (memory_config.is_sharded())
    {
        num_pages = memory_config.shard_spec.value().shape[0] * memory_config.shard_spec.value().shape[1] / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;
    };

    return page_size * num_pages;
}

uint32_t calculate_tensor_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config)
{
    if (memory_config.is_l1())
    {
        if (memory_config.is_sharded()) {
            tt::tt_metal::ShardSpecBuffer shard_spec_buffer( // this structure is not used inside validate_sharded_buffer_allocation
                                                             // assembling it with data from memory_config
                memory_config.shard_spec.value().grid,
                memory_config.shard_spec.value().shape,
                memory_config.shard_spec.value().orientation,
                memory_config.shard_spec.value().halo,
                {32, 32}, //
                {32, 32}  //
            );
            tt::tt_metal::tensor_impl::validate_sharded_buffer_allocation(shape.value, layout, data_type, shard_spec_buffer, memory_config);

            auto total_size_bytes = shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
            auto num_of_cores = memory_config.shard_spec.value().grid.num_cores();
            return total_size_bytes / num_of_cores;
        } else {
            //          Banks are ½ size of L1 (732KB)
            auto total_size_bytes = shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
            auto num_of_cores = 64; // Nebula_x1
            return total_size_bytes / num_of_cores;
        }
    }
    return (uint32_t)0; // dram not implemented yet
}

#include "binary_constraints.hpp" // for EltwiseOpConstraintsDirector::GetEltwiseOpType(..)
std::unique_ptr<EltwiseOpL1Usage> EltwiseOpL1UsageFactory::Make(const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const std::optional<EltwiseOpParams>& output)
{
    const auto input_shape_a = std::get<ttnn::types::Shape>(input_a);
    const auto memory_config_a = std::get<tt::tt_metal::MemoryConfig>(input_a);
    const auto input_shape_b = std::get<ttnn::types::Shape>(input_b);
    const auto memory_config_b = std::get<tt::tt_metal::MemoryConfig>(input_b);

    auto eltwise_op_type = EltwiseOpConstraintsFactory::GetEltwiseOpType(input_shape_a, memory_config_a, input_shape_b, memory_config_b);

    switch (eltwise_op_type) {
        case EltwiseOpTypes::ElementWiseMultiCore:
            return std::make_unique<ElementWiseMultiCoreOpL1Usage>(input_a, input_b, output);
        case EltwiseOpTypes::BroadcastWidthMultiCore:
            return std::make_unique<BroadcastWidthMultiCoreOpL1Usage>(input_a, input_b, output);
        case EltwiseOpTypes::BroadcastHeightMultiCore: // not implemented yet
        case EltwiseOpTypes::BroadcastHeightAndWidthMultiCore: // not implemented yet
        case EltwiseOpTypes::BroadcastHeightMultiCoreSharded: // not implemented yet
        case EltwiseOpTypes::BroadcastHeightMultiCoreShardedOptimized: // not implemented yet
        default:
            return nullptr;
    }
};

EltwiseOpL1Usage::EltwiseOpL1Usage(const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output) :
    input_a(input_a),
    input_b(input_b),
    output(output) {};

ElementWiseMultiCoreOpL1Usage::ElementWiseMultiCoreOpL1Usage(const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const std::optional<EltwiseOpParams>& output)
:   EltwiseOpL1Usage(input_a, input_b, output.has_value() ? output.value() : get_larger_eltwise_op_params_by_volume(input_a, input_b))
,   intermediate(calculate_intermediate_buffer_impl(input_a, input_b))
{
}

std::vector<std::tuple<uint32_t, uint32_t>> ElementWiseMultiCoreOpL1Usage::ElementWiseMultiCoreOpL1Usage::get_circular_buffer_l1_allocations_per_core() const
{
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;
    sizes.emplace_back(
        std::make_tuple(
            calculate_circular_buffer_l1_allocation_size_per_core(input_a),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_a).shard_spec)));
    sizes.emplace_back(
        std::make_tuple(
            calculate_circular_buffer_l1_allocation_size_per_core(input_b),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_b).shard_spec)));
    if (intermediate.has_value())
    {
        sizes.emplace_back(
            std::make_tuple(
                calculate_circular_buffer_l1_allocation_size_per_core(intermediate.value()),
                get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(intermediate.value()).shard_spec)));
    }
    sizes.emplace_back(
        std::make_tuple(
            calculate_circular_buffer_l1_allocation_size_per_core(output),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return std::move(sizes);
}

std::vector<std::tuple<uint32_t, uint32_t>> ElementWiseMultiCoreOpL1Usage::ElementWiseMultiCoreOpL1Usage::get_tensor_l1_allocations_per_core() const
{
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (intermediate.has_value())
    {
        sizes.emplace_back(
            std::make_tuple(
                calculate_tensor_l1_allocation_size_per_core(intermediate.value()),
                get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(intermediate.value()).shard_spec)));
    }

    sizes.emplace_back(
        std::make_tuple(
            calculate_tensor_l1_allocation_size_per_core(output),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return std::move(sizes);
}


std::optional<EltwiseOpParams> ElementWiseMultiCoreOpL1Usage::calculate_intermediate_buffer_impl(const EltwiseOpParams& input_a, const EltwiseOpParams& input_b)
{
    const auto shape_a = std::get<ttnn::types::Shape>(input_a);
    const auto shape_b = std::get<ttnn::types::Shape>(input_b);

    // internal buffer at ElementWiseMultiCore is used for ONLY batch broadcasts
    bool is_batch_broadcast = false;
    if ((shape_a.rank() == 4) && (shape_a.rank() == 4)) {
        if (shape_a[0] != shape_b[0]) {
            is_batch_broadcast = true;
        }
    }
    if (!is_batch_broadcast)
    {
        return std::nullopt;
    }

    auto intermediate = (shape_a[0] > shape_b[0]) ? input_b : input_a;
    assert(std::get<ttnn::types::Shape>(intermediate).rank() == 4); // my implementation limitation

    auto batch_size = (shape_a[0] > shape_b[0]) ? shape_a[0] : shape_b[0];;
    vector<uint32_t> new_shape;
    new_shape.push_back(batch_size);
    for (int i = 1; i < 4; i++) {
        new_shape.push_back(std::get<ttnn::types::Shape>(intermediate)[i]);
    }

    std::get<ttnn::types::Shape>(intermediate) = ttnn::Shape{tt::tt_metal::Shape{new_shape, tt::tt_metal::Padding{std::get<ttnn::types::Shape>(intermediate).rank()}}};

    return std::make_optional(intermediate);
}

BroadcastWidthMultiCoreOpL1Usage::BroadcastWidthMultiCoreOpL1Usage(const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const std::optional<EltwiseOpParams>& output)
:   EltwiseOpL1Usage(input_a, input_b, output.has_value() ? output.value() : get_larger_eltwise_op_params_by_volume(input_a, input_b))
,   intermediate(calculate_intermediate_buffer_impl(input_a, input_b))
{
}

std::vector<std::tuple<uint32_t, uint32_t>> BroadcastWidthMultiCoreOpL1Usage::get_circular_buffer_l1_allocations_per_core() const
{
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    sizes.emplace_back(
        std::make_tuple(
            calculate_circular_buffer_l1_allocation_size_per_core(input_a),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_a).shard_spec)));
    sizes.emplace_back(
        std::make_tuple(
            calculate_circular_buffer_l1_allocation_size_per_core(input_b),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_b).shard_spec)));
    if (intermediate.has_value()) {
        sizes.emplace_back(
            std::make_tuple(
                calculate_circular_buffer_l1_allocation_size_per_core(intermediate.value()),
                get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(intermediate.value()).shard_spec)));
    }
    sizes.emplace_back(
        std::make_tuple(
            calculate_circular_buffer_l1_allocation_size_per_core(output),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return std::move(sizes);
}

std::vector<std::tuple<uint32_t, uint32_t>> BroadcastWidthMultiCoreOpL1Usage::get_tensor_l1_allocations_per_core() const
{
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (intermediate.has_value()) {
        sizes.emplace_back(
            std::make_tuple(
                calculate_tensor_l1_allocation_size_per_core(intermediate.value()),
                get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(intermediate.value()).shard_spec)));
    }
    sizes.emplace_back(
        std::make_tuple(
            calculate_tensor_l1_allocation_size_per_core(output),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return std::move(sizes);
}

std::optional<EltwiseOpParams> BroadcastWidthMultiCoreOpL1Usage::calculate_intermediate_buffer_impl(const EltwiseOpParams& input_a, const EltwiseOpParams& input_b)
{
    const auto shape_a = std::get<ttnn::types::Shape>(input_a);
    const auto shape_b = std::get<ttnn::types::Shape>(input_b);

    // internal buffer used for batch broadcasts
    bool is_batch_broadcast = false;
    if ((shape_a.rank() == 4) && (shape_a.rank() == 4)) {
        if (shape_a[0] != shape_b[0]) {
            is_batch_broadcast = true;
        }
    }
    if (!is_batch_broadcast)
    {
        return std::nullopt;
    }

    auto intermediate = (shape_a[0] > shape_b[0]) ? input_b : input_a;
    assert(std::get<ttnn::types::Shape>(intermediate).rank() == 4); // my implementation limitation

    auto batch_size = (shape_a[0] > shape_b[0]) ? shape_a[0] : shape_b[0];;
    vector<uint32_t> new_shape;
    new_shape.push_back(batch_size);
    for (int i = 1; i < 4; i++) {
        new_shape.push_back(std::get<ttnn::types::Shape>(intermediate)[i]);
    }

    std::get<ttnn::types::Shape>(intermediate) = ttnn::Shape{tt::tt_metal::Shape{new_shape, tt::tt_metal::Padding{std::get<ttnn::types::Shape>(intermediate).rank()}}};

    return std::make_optional(intermediate);
}
