#include "unary_l1_interface.hpp"

#include <memory>
#include <optional>

#include "ttnn/operations/common/l1_interface_common.hpp"

std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core_unary_impl(
    const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output) {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    const auto get_operand_cb_size = [](const L1InterfaceOperandParams& operand) {
        return is_sharded(operand)
                   ? c_cb_shares_space_with_sharded_operand
                   : calculate_circular_buffer_l1_allocation_size_per_core(operand, 1 /* max_block_size */);
    };

    sizes.emplace_back(std::make_tuple(
        get_operand_cb_size(input), get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(input).shard_spec)));

    sizes.emplace_back(std::make_tuple(
        get_operand_cb_size(output), get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core_unary_impl(
    const L1InterfaceOperandParams& output) {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    sizes.emplace_back(std::make_tuple(
        calculate_tensor_l1_allocation_size_per_core(output),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return sizes;
}

std::unique_ptr<UnaryOpL1Usage> UnaryOpL1UsageFactory::Make(
    const L1InterfaceOperandParams& input, const std::optional<L1InterfaceOperandParams>& output) {
    const auto input_memory_config = std::get<tt::tt_metal::MemoryConfig>(input);

    if (input_memory_config.is_sharded()) {
        return std::make_unique<ShardedUnaryOpL1Usage>(input, output.value_or(input));
    } else {
        return std::make_unique<InterleavedUnaryOpL1Usage>(input, output.value_or(input));
    }
};

UnaryOpL1Usage::UnaryOpL1Usage(const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output) :
    input(input), output(output) {}

InterleavedUnaryOpL1Usage::InterleavedUnaryOpL1Usage(
    const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output) :
    UnaryOpL1Usage(input, output) {}

std::vector<std::tuple<uint32_t, uint32_t>>
InterleavedUnaryOpL1Usage::InterleavedUnaryOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    return get_circular_buffer_l1_allocations_per_core_unary_impl(input, output);
}

std::vector<std::tuple<uint32_t, uint32_t>>
InterleavedUnaryOpL1Usage::InterleavedUnaryOpL1Usage::get_tensor_l1_allocations_per_core() const {
    return get_tensor_l1_allocations_per_core_unary_impl(output);
}

ShardedUnaryOpL1Usage::ShardedUnaryOpL1Usage(
    const L1InterfaceOperandParams& input, const L1InterfaceOperandParams& output) :
    UnaryOpL1Usage(input, output) {}

std::vector<std::tuple<uint32_t, uint32_t>>
ShardedUnaryOpL1Usage::ShardedUnaryOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    return get_circular_buffer_l1_allocations_per_core_unary_impl(input, output);
}

std::vector<std::tuple<uint32_t, uint32_t>>
ShardedUnaryOpL1Usage::ShardedUnaryOpL1Usage::get_tensor_l1_allocations_per_core() const {
    return get_tensor_l1_allocations_per_core_unary_impl(output);
}
