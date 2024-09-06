#include "binary_l1_interface.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "detail/util.hpp"
#include "impl/buffers/buffer.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor_impl.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"

uint32_t get_num_of_cores(const std::optional<tt::tt_metal::ShardSpec>& shard_spec) {
    if (shard_spec.has_value()) {
        return shard_spec.value().grid.num_cores();
    }
    return (uint32_t)64;  // Nebula_x1
};

uint32_t get_num_pages(const tt::tt_metal::ShardSpec& shard_spec) {
    return shard_spec.shape[0] * shard_spec.shape[1] / tt::constants::TILE_HW;
}

EltwiseOpParams get_larger_eltwise_op_params_by_volume(const EltwiseOpParams& a, const EltwiseOpParams& b) {
    if (std::get<ttnn::types::Shape>(a).volume() > std::get<ttnn::types::Shape>(b).volume()) {
        return a;
    } else {
        return b;
    }
};

// these should go to a common tensor_l1_interface header or something
uint32_t calculate_circular_buffer_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const uint32_t max_block_size) {
    auto total_size_bytes =
        shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
    auto page_size = tt::tt_metal::tensor_impl::get_page_size(data_type, layout, total_size_bytes, shape.value);
    auto num_pages = memory_config.is_sharded() ? get_num_pages(memory_config.shard_spec.value()) : 2 * max_block_size;

    return page_size * num_pages;
}

uint32_t calculate_repeat_circular_buffer_size(tt::tt_metal::DataType data_type) {
    const auto repeat_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(data_type);

    return 2 * tt::tt_metal::detail::TileSize(repeat_cb_data_format);
}

uint32_t calculate_max_block_size(const std::optional<tt::tt_metal::ShardSpec>& shard_spec) {
    if (!shard_spec.has_value()) {
        return (uint32_t)1;
    }

    const uint32_t num_tiles_per_shard = get_num_pages(shard_spec.value());
    const uint32_t max_block_size = 8;

    for (int find_divisor = max_block_size; find_divisor >= 1; find_divisor--) {
        if (num_tiles_per_shard % find_divisor == 0) {
            return find_divisor;
        }
    }

    return (uint32_t)1;
}

uint32_t calculate_tensor_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config) {
    if (memory_config.is_l1()) {
        if (memory_config.is_sharded()) {
            tt::tt_metal::ShardSpecBuffer
                shard_spec_buffer(  // this structure is not used inside validate_sharded_buffer_allocation
                                    // assembling it with data from memory_config
                    memory_config.shard_spec.value().grid,
                    memory_config.shard_spec.value().shape,
                    memory_config.shard_spec.value().orientation,
                    memory_config.shard_spec.value().halo,
                    {32, 32},  //
                    {32, 32}   //
                );
            tt::tt_metal::tensor_impl::validate_sharded_buffer_allocation(
                shape.value, layout, data_type, shard_spec_buffer, memory_config);

            auto total_size_bytes =
                shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
            auto num_of_cores = memory_config.shard_spec.value().grid.num_cores();
            return total_size_bytes / num_of_cores;
        } else {
            //          Banks are ½ size of L1 (732KB)
            auto total_size_bytes =
                shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
            auto num_of_cores = 64;  // Nebula_x1
            return total_size_bytes / num_of_cores;
        }
    }
    return (uint32_t)0;  // dram not implemented yet
}

#include "binary_constraints.hpp"  // for EltwiseOpConstraintsDirector::GetEltwiseOpType(..)
std::unique_ptr<EltwiseOpL1Usage> EltwiseOpL1UsageFactory::Make(
    const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output) {
    const auto input_shape_a = std::get<ttnn::types::Shape>(input_a);
    const auto memory_config_a = std::get<tt::tt_metal::MemoryConfig>(input_a);
    const auto input_shape_b = std::get<ttnn::types::Shape>(input_b);
    const auto memory_config_b = std::get<tt::tt_metal::MemoryConfig>(input_b);
    const auto memory_config_o = std::get<tt::tt_metal::MemoryConfig>(output);

    auto eltwise_op_type = EltwiseOpConstraintsFactory::GetEltwiseOpType(
        input_shape_a, memory_config_a, input_shape_b, memory_config_b, memory_config_o);

    switch (eltwise_op_type) {
        case EltwiseOpTypes::ElementWiseMultiCore:
            return std::make_unique<ElementWiseMultiCoreOpL1Usage>(input_a, input_b, output);
        case EltwiseOpTypes::BroadcastWidthMultiCore:
            return std::make_unique<BroadcastWidthMultiCoreOpL1Usage>(input_a, input_b, output);
        case EltwiseOpTypes::BroadcastHeightMultiCore:                  // not implemented yet
        case EltwiseOpTypes::BroadcastHeightAndWidthMultiCore:          // not implemented yet
        case EltwiseOpTypes::BroadcastHeightMultiCoreSharded:           // not implemented yet
        case EltwiseOpTypes::BroadcastHeightMultiCoreShardedOptimized:  // not implemented yet
        default: return nullptr;
    }
};

EltwiseOpL1Usage::EltwiseOpL1Usage(
    const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output) :
    input_a(input_a), input_b(input_b), output(output), repeat(calculate_repeat_buffer_impl(input_a, input_b)){};

std::optional<EltwiseOpParams> EltwiseOpL1Usage::calculate_repeat_buffer_impl(
    const EltwiseOpParams& input_a, const EltwiseOpParams& input_b) {
    const auto shape_a = std::get<ttnn::types::Shape>(input_a);
    const auto shape_b = std::get<ttnn::types::Shape>(input_b);

    bool is_batch_broadcast = false;
    if ((shape_a.rank() == 4) && (shape_a.rank() == 4)) {
        if (shape_a[0] != shape_b[0]) {
            is_batch_broadcast = true;
        }
    }
    if (!is_batch_broadcast) {
        return std::nullopt;
    }

    auto intermediate = (shape_a[0] > shape_b[0]) ? input_b : input_a;
    assert(std::get<ttnn::types::Shape>(intermediate).rank() == 4);  // my implementation limitation

    auto batch_size = (shape_a[0] > shape_b[0]) ? shape_a[0] : shape_b[0];
    ;
    vector<uint32_t> new_shape;
    new_shape.push_back(batch_size);
    for (int i = 1; i < 4; i++) {
        new_shape.push_back(std::get<ttnn::types::Shape>(intermediate)[i]);
    }

    std::get<ttnn::types::Shape>(intermediate) = ttnn::Shape{
        tt::tt_metal::Shape{new_shape, tt::tt_metal::Padding{std::get<ttnn::types::Shape>(intermediate).rank()}}};

    return std::make_optional(intermediate);
}

std::optional<ShardSpec> EltwiseOpL1Usage::get_op_shard_spec() const {
    const auto memory_config_a = std::get<tt::tt_metal::MemoryConfig>(input_a);
    const auto memory_config_b = std::get<tt::tt_metal::MemoryConfig>(input_b);
    const auto memory_config_output = std::get<tt::tt_metal::MemoryConfig>(output);

    std::optional<ShardSpec> op_shard_spec = std::nullopt;
    if (memory_config_a.is_sharded()) {
        op_shard_spec = memory_config_a.shard_spec;
    } else if (memory_config_b.is_sharded()) {
        op_shard_spec = memory_config_b.shard_spec;
    } else if (memory_config_output.is_sharded()) {
        op_shard_spec = memory_config_output.shard_spec;
    }

    return op_shard_spec;
}

ElementWiseMultiCoreOpL1Usage::ElementWiseMultiCoreOpL1Usage(
    const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output) :
    EltwiseOpL1Usage(input_a, input_b, output) {}

std::vector<std::tuple<uint32_t, uint32_t>>
ElementWiseMultiCoreOpL1Usage::ElementWiseMultiCoreOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;
    if (repeat.has_value()) {
        sizes.emplace_back(std::make_tuple(
            calculate_repeat_circular_buffer_size(std::get<tt::tt_metal::DataType>(repeat.value())),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(repeat.value()).shard_spec)));
    }

    const uint32_t max_block_size = calculate_max_block_size(get_op_shard_spec());

    sizes.emplace_back(std::make_tuple(
        calculate_circular_buffer_l1_allocation_size_per_core(input_a, max_block_size),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_a).shard_spec)));
    sizes.emplace_back(std::make_tuple(
        calculate_circular_buffer_l1_allocation_size_per_core(input_b, max_block_size),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_b).shard_spec)));
    sizes.emplace_back(std::make_tuple(
        calculate_circular_buffer_l1_allocation_size_per_core(output, max_block_size),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>>
ElementWiseMultiCoreOpL1Usage::ElementWiseMultiCoreOpL1Usage::get_tensor_l1_allocations_per_core() const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (repeat.has_value()) {
        sizes.emplace_back(std::make_tuple(
            calculate_tensor_l1_allocation_size_per_core(repeat.value()),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(repeat.value()).shard_spec)));
    }

    sizes.emplace_back(std::make_tuple(
        calculate_tensor_l1_allocation_size_per_core(output),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return sizes;
}

BroadcastWidthMultiCoreOpL1Usage::BroadcastWidthMultiCoreOpL1Usage(
    const EltwiseOpParams& input_a, const EltwiseOpParams& input_b, const EltwiseOpParams& output) :
    EltwiseOpL1Usage(input_a, input_b, output) {}

std::vector<std::tuple<uint32_t, uint32_t>>
BroadcastWidthMultiCoreOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (repeat.has_value()) {
        sizes.emplace_back(std::make_tuple(
            calculate_repeat_circular_buffer_size(std::get<tt::tt_metal::DataType>(repeat.value())),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(repeat.value()).shard_spec)));
    }

    const uint32_t max_block_size = calculate_max_block_size(get_op_shard_spec());

    sizes.emplace_back(std::make_tuple(
        calculate_circular_buffer_l1_allocation_size_per_core(input_a, max_block_size),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_a).shard_spec)));
    sizes.emplace_back(std::make_tuple(
        calculate_circular_buffer_l1_allocation_size_per_core(input_b, max_block_size),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_b).shard_spec)));
    sizes.emplace_back(std::make_tuple(
        calculate_circular_buffer_l1_allocation_size_per_core(output, max_block_size),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>> BroadcastWidthMultiCoreOpL1Usage::get_tensor_l1_allocations_per_core()
    const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (repeat.has_value()) {
        sizes.emplace_back(std::make_tuple(
            calculate_tensor_l1_allocation_size_per_core(repeat.value()),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(repeat.value()).shard_spec)));
    }

    sizes.emplace_back(std::make_tuple(
        calculate_tensor_l1_allocation_size_per_core(output),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return sizes;
}
