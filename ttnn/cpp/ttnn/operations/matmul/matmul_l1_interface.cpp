#include "matmul_l1_interface.hpp"

#include <cassert>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

#include "common/constants.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"

MatmulOPL1Usage::MatmulOPL1Usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output) :
    input_a(input_a), input_b(input_b), output(output) {}

MatmulMultiCoreReuseMultiCastOpL1Usage::MatmulMultiCoreReuseMultiCastOpL1Usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& program_config) :
    MatmulOPL1Usage(input_a, input_b, output), program_config(program_config) {}

std::vector<std::tuple<uint32_t, uint32_t>>
MatmulMultiCoreReuseMultiCastOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    uint32_t B = get_batch_size(get_legacy_shape(input_a));
    uint32_t M = get_legacy_shape(input_a)[-2] / tt::constants::TILE_HEIGHT;
    uint32_t N = get_legacy_shape(input_b)[-1] / tt::constants::TILE_WIDTH;
    uint32_t K = get_legacy_shape(input_a)[-1] / tt::constants::TILE_WIDTH;

    if (program_config.fuse_batch) {
        M = B * M;
        B = 1;
    }

    uint32_t num_blocks = K / program_config.in0_block_w;
    const uint32_t input_cb_size_multiplier = B * num_blocks > 1 ? 2 : 1;

    uint32_t in0_CB_size = has_layout(input_a, tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED)
                               ? c_cb_shares_space_with_sharded_operand
                               : input_cb_size_multiplier * program_config.per_core_M * program_config.in0_block_w *
                                     get_tile_size(input_a);

    uint32_t in1_CB_size = is_sharded(input_b) && !std::get<tt::tt_metal::MemoryConfig>(input_b).is_dram()
                               ? c_cb_shares_space_with_sharded_operand
                               : input_cb_size_multiplier * program_config.per_core_N * program_config.in0_block_w *
                                     get_tile_size(input_b);

    uint32_t out_block_size = is_sharded(output)
                                  ? c_cb_shares_space_with_sharded_operand
                                  : program_config.per_core_M * program_config.per_core_N * get_tile_size(output);

    sizes.push_back(std::make_tuple(in0_CB_size, get_num_of_cores()));
    sizes.push_back(std::make_tuple(in1_CB_size, get_num_of_cores()));
    if (has_layout(input_a, TensorMemoryLayout::BLOCK_SHARDED)) {
        sizes.push_back(std::make_tuple(c_cb_shares_space_with_sharded_operand, get_num_of_cores()));
        sizes.push_back(std::make_tuple(2 * 32, get_num_of_cores()));
    }

    sizes.push_back(std::make_tuple(out_block_size, get_num_of_cores()));

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>> MatmulMultiCoreReuseMultiCastOpL1Usage::get_tensor_l1_allocations_per_core()
    const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (!std::get<tt::tt_metal::MemoryConfig>(output).is_dram()) {
        sizes.emplace_back(std::make_tuple(
            calculate_tensor_l1_allocation_size_per_core(output),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));
    }

    return sizes;
}

MatmulMultiCoreReuseMultiCast1DOpL1Usage::MatmulMultiCoreReuseMultiCast1DOpL1Usage(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config) :
    MatmulOPL1Usage(input_a, input_b, output), program_config(program_config) {}

std::vector<std::tuple<uint32_t, uint32_t>>
MatmulMultiCoreReuseMultiCast1DOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    uint32_t B = get_batch_size(get_legacy_shape(input_a));
    uint32_t M = get_legacy_shape(input_a)[-2] / tt::constants::TILE_HEIGHT;
    uint32_t N = get_legacy_shape(input_b)[-1] / tt::constants::TILE_WIDTH;
    uint32_t K = get_legacy_shape(input_a)[-1] / tt::constants::TILE_WIDTH;
    uint32_t num_blocks = K / program_config.in0_block_w;
    if (program_config.fuse_batch) {
        M = B * M;
        B = 1;
    }

    if (program_config.mcast_in0) {
        return get_circular_buffer_l1_allocations_per_core_mcast_in0(B, M, N, K, num_blocks);
    } else {
        return get_circular_buffer_l1_allocations_per_core_mcast_in1(B, M, N, K, num_blocks);
    }
}

std::vector<std::tuple<uint32_t, uint32_t>>
MatmulMultiCoreReuseMultiCast1DOpL1Usage::get_circular_buffer_l1_allocations_per_core_mcast_in0(
    const uint32_t B, const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t num_blocks) const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    const uint32_t input_cb_size_multiplier = B * num_blocks > 1 ? 2 : 1;
    uint32_t in0_CB_size =
        input_cb_size_multiplier * program_config.per_core_M * program_config.in0_block_w * get_tile_size(input_a);
    uint32_t in1_CB_size =
        input_cb_size_multiplier * program_config.per_core_N * program_config.in0_block_w * get_tile_size(input_b);

    uint32_t out_CB_size = is_sharded(output)
                               ? c_cb_shares_space_with_sharded_operand
                               : program_config.per_core_M * program_config.per_core_N * get_tile_size(output);

    sizes.push_back(std::make_tuple(in0_CB_size, get_num_of_cores()));
    sizes.push_back(std::make_tuple(in1_CB_size, get_num_of_cores()));
    if (is_sharded(input_a)) {
        sizes.push_back(std::make_tuple(c_cb_shares_space_with_sharded_operand, get_num_of_cores()));
        sizes.push_back(std::make_tuple(32 * 2, get_num_of_cores()));
    }
    sizes.push_back(std::make_tuple(out_CB_size, get_num_of_cores()));

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>>
MatmulMultiCoreReuseMultiCast1DOpL1Usage::get_circular_buffer_l1_allocations_per_core_mcast_in1(
    const uint32_t B, const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t num_blocks) const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    const uint32_t input_cb_size_multiplier = B * num_blocks > 1 ? 2 : 1;
    const auto in0_shard_spec_opt = get_shard_spec(input_a);
    bool extract_shard_sub_blocks =
        is_sharded(input_a) &&
        ((in0_shard_spec_opt.value().shape[1] / tt::constants::TILE_WIDTH) / program_config.in0_block_w > 1);

    uint32_t in0_CB_tiles = is_sharded(input_a)
                                ? num_blocks * program_config.per_core_M * program_config.in0_block_w * B
                                : input_cb_size_multiplier * program_config.per_core_M * program_config.in0_block_w;
    uint32_t in0_CB_size = is_sharded(input_a) && extract_shard_sub_blocks ? c_cb_shares_space_with_sharded_operand
                                                                           : in0_CB_tiles * get_tile_size(input_a);

    uint32_t in1_CB_size = is_sharded(input_b) && !std::get<tt::tt_metal::MemoryConfig>(input_b).is_dram()
                               ? c_cb_shares_space_with_sharded_operand
                               : input_cb_size_multiplier * program_config.per_core_N * program_config.in0_block_w *
                                     get_tile_size(input_b);

    uint32_t out_CB_size = is_sharded(output)
                               ? c_cb_shares_space_with_sharded_operand
                               : program_config.per_core_M * program_config.per_core_N * get_tile_size(output);

    sizes.push_back(std::make_tuple(in0_CB_size, get_num_of_cores()));
    if (is_sharded(input_a) && extract_shard_sub_blocks) {
        sizes.push_back(std::make_tuple(c_cb_shares_space_with_sharded_operand, get_num_of_cores()));
    }
    sizes.push_back(std::make_tuple(in1_CB_size, get_num_of_cores()));
    sizes.push_back(std::make_tuple(out_CB_size, get_num_of_cores()));

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>>
MatmulMultiCoreReuseMultiCast1DOpL1Usage::get_tensor_l1_allocations_per_core() const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (!std::get<tt::tt_metal::MemoryConfig>(output).is_dram()) {
        sizes.emplace_back(std::make_tuple(
            calculate_tensor_l1_allocation_size_per_core(output),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));
    }

    return sizes;
}

std::unique_ptr<MatmulOPL1Usage> MatmulOpL1UsageFactory::Make(
    const L1InterfaceOperandParams& input_a,
    const L1InterfaceOperandParams& input_b,
    const L1InterfaceOperandParams& output,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config) {
    std::unique_ptr<MatmulOPL1Usage> l1_usage = nullptr;
    std::visit(
        [&](const auto& program_config) {
            using T = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<T, ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                l1_usage =
                    std::make_unique<MatmulMultiCoreReuseMultiCastOpL1Usage>(input_a, input_b, output, program_config);
            } else if constexpr (std::is_same_v<
                                     T,
                                     ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                l1_usage = std::make_unique<MatmulMultiCoreReuseMultiCast1DOpL1Usage>(
                    input_a, input_b, output, program_config);
            }
        },
        program_config);

    return l1_usage;
}
