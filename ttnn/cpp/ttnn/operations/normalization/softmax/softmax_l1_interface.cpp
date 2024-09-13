#include "softmax_l1_interface.hpp"

#include <cstdint>
#include <memory>
#include <tuple>

#include "common/constants.hpp"
#include "detail/util.hpp"
#include "impl/buffers/buffer.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/tensor/types.hpp"

L1InterfaceOperandParams get_softmax_output(const L1InterfaceOperandParams& input) {
    return std::make_tuple(
        std::get<ttnn::types::Shape>(input),
        std::get<tt::tt_metal::DataType>(input),
        tt::tt_metal::Layout::TILE,
        std::get<tt::tt_metal::MemoryConfig>(input));
}

// Copied over from ttnn work split.
int find_max_divisor(uint32_t val, uint32_t start_max_div) {
    int result = 1;
    for (int find_divisor = start_max_div; find_divisor >= 1; find_divisor--) {
        if (find_divisor == 7 || find_divisor == 5)
            continue;
        if (val % find_divisor == 0) {
            result = find_divisor;
            break;
        }
    }
    return result;
}

std::unique_ptr<SoftmaxOpL1Usage> SoftmaxOpL1UsageFactory::Make(const L1InterfaceOperandParams& input, int dim_arg) {
    return std::make_unique<SoftmaxOpL1Usage>(input, dim_arg);
};

SoftmaxOpL1Usage::SoftmaxOpL1Usage(const L1InterfaceOperandParams& input, int dim_arg) :
    input(input), output(get_softmax_output(input)), dim_arg(dim_arg), block_size(calculate_block_size_impl(input)) {}

bool SoftmaxOpL1Usage::should_tilize_input() const {
    return std::get<tt::tt_metal::Layout>(input) != tt::tt_metal::Layout::TILE;
}

uint32_t SoftmaxOpL1Usage::calculate_block_size_impl(const L1InterfaceOperandParams& input) const {
    const auto shape = std::get<ttnn::types::Shape>(input).value;
    const auto input_volume = std::get<ttnn::types::Shape>(input).volume();

    uint32_t num_tiles = input_volume / tt::constants::TILE_HW;
    uint32_t Wt = shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = (input_volume / (shape[0] * shape[-1])) / tt::constants::TILE_HEIGHT;
    uint32_t block_size = find_max_divisor(Wt, 8);

    return block_size;
}

uint32_t SoftmaxOpL1Usage::get_input_cb_size() const {
    return is_sharded(input) ? c_cb_shares_space_with_sharded_operand : 2 * block_size * get_tile_size(input);
}

uint32_t SoftmaxOpL1Usage::get_output_cb_size() const {
    return is_sharded(output) ? c_cb_shares_space_with_sharded_operand : 2 * block_size * get_tile_size(output);
}

std::vector<uint32_t> SoftmaxOpL1Usage::get_intermediate_cb_sizes() const {
    uint32_t im1_t = 1;  // 1/sum(exp(x))
    uint32_t in2_t = 1;  // scaler for reduce coming from reader
    uint32_t in5_t = 1;
    uint32_t im0_t =
        block_size * tt::div_up(std::get<ttnn::Shape>(input).value[-1] / tt::constants::TILE_WIDTH, block_size);

    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    // TODO: Support fp32 accumulation for WH B0 arch
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    return {im1_t * im_tile_size, in2_t * scalar_tile_size, im0_t * im_tile_size, in5_t * mask_tile_size};
}

uint32_t SoftmaxOpL1Usage::get_tilize_cb_size() const {
    const auto shape = std::get<ttnn::types::Shape>(input).value;
    const auto input_volume = std::get<ttnn::types::Shape>(input).volume();
    uint32_t stick_s = shape[-1];
    uint32_t num_sticks = input_volume / shape[-1];

    uint32_t num_tiles_in_row = stick_s / tt::constants::TILE_WIDTH;
    // TODO: Replace this with actual device L1 space query
    uint32_t max_l1_size = 732 * 1024;
    uint32_t max_tiles = max_l1_size / 2 * get_tile_size(input);  // 2 CBs
    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }

    return num_tiles_per_block * get_tile_size(input);
}

std::vector<std::tuple<uint32_t, uint32_t>> SoftmaxOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (should_tilize_input()) {
        const uint32_t tilize_cb_size = get_tilize_cb_size();
        sizes.push_back(std::make_tuple(tilize_cb_size, (uint32_t)1));
        sizes.push_back(std::make_tuple(tilize_cb_size, (uint32_t)1));
    }

    const uint32_t num_cores_with_storage = get_num_of_cores();
    sizes.push_back(std::make_tuple(get_input_cb_size(), num_cores_with_storage));
    sizes.push_back(std::make_tuple(get_output_cb_size(), num_cores_with_storage));
    for (const uint32_t intermediate_cb_size : get_intermediate_cb_sizes()) {
        sizes.push_back(std::make_tuple(intermediate_cb_size, num_cores_with_storage));
    }

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>> SoftmaxOpL1Usage::get_tensor_l1_allocations_per_core() const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    if (should_tilize_input()) {
        sizes.emplace_back(std::make_tuple(
            calculate_tensor_l1_allocation_size_per_core(output),
            get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));
    }

    sizes.emplace_back(std::make_tuple(
        calculate_tensor_l1_allocation_size_per_core(output),
        get_num_of_cores(std::get<tt::tt_metal::MemoryConfig>(output).shard_spec)));

    return sizes;
}
