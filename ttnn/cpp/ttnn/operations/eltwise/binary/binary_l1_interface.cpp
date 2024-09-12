#include "binary_l1_interface.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "ttnn/operations/common/l1_interface_common.hpp"

L1InterfaceOperandParams get_larger_eltwise_op_params_by_volume(
    const L1InterfaceOperandParams& a, const L1InterfaceOperandParams& b) {
    if (std::get<ttnn::types::Shape>(a).volume() > std::get<ttnn::types::Shape>(b).volume()) {
        return a;
    } else {
        return b;
    }
};

std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core_eltwise_impl(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const std::optional<L1InterfaceOperandParams>& repeat = std::nullopt,
    const std::optional<ShardSpec>& op_shard_spec = std::nullopt) {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;
    if (repeat.has_value()) {
        sizes.emplace_back(std::make_tuple(
            calculate_repeat_circular_buffer_size(std::get<tt::tt_metal::DataType>(repeat.value())),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(repeat.value()).shard_spec)));
    }

    const uint32_t max_block_size = calculate_max_block_size(op_shard_spec);

    const auto get_operand_cb_size = [max_block_size](const L1InterfaceOperandParams& operand) {
        return is_sharded(operand) ? c_cb_shares_space_with_sharded_operand
                                   : calculate_circular_buffer_l1_allocation_size_per_core(operand, max_block_size);
    };

    sizes.emplace_back(std::make_tuple(
        get_operand_cb_size(input_a), get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_a).shard_spec)));
    sizes.emplace_back(std::make_tuple(
        get_operand_cb_size(input_b), get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input_b).shard_spec)));
    sizes.emplace_back(std::make_tuple(
        get_operand_cb_size(output), get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core_eltwise_impl(
    const L1InterfaceOperandParams& output, const std::optional<L1InterfaceOperandParams>& repeat = std::nullopt) {
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

#include "binary_constraints.hpp"  // for EltwiseOpConstraintsDirector::GetEltwiseOpType(..)
std::unique_ptr<EltwiseOpL1Usage> EltwiseOpL1UsageFactory::Make(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) {
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
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) :
    input_a(input_a), input_b(input_b), output(output), repeat(calculate_repeat_buffer_impl(input_a, input_b)){};

std::optional<L1InterfaceOperandParams> EltwiseOpL1Usage::calculate_repeat_buffer_impl(
    const L1InterfaceOperandParams& input_a, const L1InterfaceOperandParams& input_b) {
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
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) :
    EltwiseOpL1Usage(input_a, input_b, output) {}

std::vector<std::tuple<uint32_t, uint32_t>>
ElementWiseMultiCoreOpL1Usage::ElementWiseMultiCoreOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    return get_circular_buffer_l1_allocations_per_core_eltwise_impl(
        input_a, input_b, output, repeat, get_op_shard_spec());
}

std::vector<std::tuple<uint32_t, uint32_t>> ElementWiseMultiCoreOpL1Usage::get_tensor_l1_allocations_per_core() const {
    return get_tensor_l1_allocations_per_core_eltwise_impl(output, repeat);
}

BroadcastWidthMultiCoreOpL1Usage::BroadcastWidthMultiCoreOpL1Usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) :
    EltwiseOpL1Usage(input_a, input_b, output) {}

std::vector<std::tuple<uint32_t, uint32_t>>
BroadcastWidthMultiCoreOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    return get_circular_buffer_l1_allocations_per_core_eltwise_impl(
        input_a, input_b, output, repeat, get_op_shard_spec());
}

std::vector<std::tuple<uint32_t, uint32_t>> BroadcastWidthMultiCoreOpL1Usage::get_tensor_l1_allocations_per_core()
    const {
    return get_tensor_l1_allocations_per_core_eltwise_impl(output, repeat);
}
