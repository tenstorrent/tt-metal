#include "softmax_l1_interface.hpp"

#include <cstdint>
#include <memory>
#include <tuple>

#include "detail/util.hpp"
#include "ttnn/operations/eltwise/common/eltwise_l1_interface_common.hpp"
#include "ttnn/tensor/types.hpp"

EltwiseOpParams get_softmax_output(const EltwiseOpParams& input) {
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

std::unique_ptr<SoftmaxOpL1Usage> SoftmaxOpL1UsageFactory::Make(const EltwiseOpParams& input, int dim_arg) {
    return std::make_unique<SoftmaxOpL1Usage>(input, dim_arg);
};

SoftmaxOpL1Usage::SoftmaxOpL1Usage(const EltwiseOpParams& input, int dim_arg) :
    input(input), output(get_softmax_output(input)), dim_arg(dim_arg) {}

bool SoftmaxOpL1Usage::SoftmaxOpL1Usage::should_tilize_input() const {
    return std::get<tt::tt_metal::Layout>(input) != tt::tt_metal::Layout::TILE;
}

std::vector<std::tuple<uint32_t, uint32_t>>
SoftmaxOpL1Usage::SoftmaxOpL1Usage::get_circular_buffer_l1_allocations_per_core() const {
    std::vector<std::tuple<uint32_t, uint32_t>> sizes;

    const auto shape = std::get<ttnn::types::Shape>(input).value;
    const auto input_volume = std::get<ttnn::types::Shape>(input).volume();

    uint32_t W = shape[-1], H = (input_volume / (shape[0] * shape[-1])), NC = shape[0];
    uint32_t HW = H * W;
    uint32_t num_tiles = input_volume / tt::constants::TILE_HW;
    uint32_t Wt = W / tt::constants::TILE_WIDTH;
    uint32_t Ht = H / tt::constants::TILE_HEIGHT;
    uint32_t block_size = false ? find_max_divisor(Wt, 4) : find_max_divisor(Wt, 8);

    uint32_t in0_t = block_size * 2;
    uint32_t out0_t = block_size * 2;
    uint32_t im1_t = 1;  // 1/sum(exp(x))
    uint32_t in2_t = 1;  // scaler for reduce coming from reader
    uint32_t in3_t = 1;  // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in4_t =
        tt::div_up(Wt, block_size) * block_size;  // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled
    uint32_t in5_t = 1;
    uint32_t im0_t = block_size * tt::div_up(Wt, block_size);
    uint32_t im3_t = block_size * (tt::div_up(Wt, block_size) + 1);

    uint32_t in0_tile_size = tt::tt_metal::detail::TileSize(
        tt::tt_metal::datatype_to_dataformat_converter(std::get<tt::tt_metal::DataType>(input)));
    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t out0_tile_size = tt::tt_metal::detail::TileSize(
        tt::tt_metal::datatype_to_dataformat_converter(std::get<tt::tt_metal::DataType>(output)));
    uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    // TODO: Support fp32 accumulation for WH B0 arch
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    if (should_tilize_input()) {
        uint32_t stick_s = W;
        uint32_t num_sticks = input_volume / W;

        uint32_t num_tiles_in_row = stick_s / tt::constants::TILE_WIDTH;
        // TODO: Replace this with actual device L1 space query
        uint32_t max_l1_size = 732 * 1024;
        uint32_t max_tiles = max_l1_size / (in0_tile_size + out0_tile_size);  // 2 CBs
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

        sizes.push_back(std::make_tuple(num_tiles_per_block * in0_tile_size, (uint32_t)1));
        sizes.push_back(std::make_tuple(num_tiles_per_block * out0_tile_size, (uint32_t)1));
    }

    const uint32_t num_cores_with_storage = get_num_of_cores();
    sizes.push_back(std::make_tuple(in0_t * in0_tile_size, num_cores_with_storage));
    sizes.push_back(std::make_tuple(out0_t * out0_tile_size, num_cores_with_storage));
    sizes.push_back(std::make_tuple(im1_t * im_tile_size, num_cores_with_storage));
    sizes.push_back(std::make_tuple(in2_t * scalar_tile_size, num_cores_with_storage));
    sizes.push_back(std::make_tuple(im0_t * im_tile_size, num_cores_with_storage));
    sizes.push_back(std::make_tuple(in5_t * mask_tile_size, num_cores_with_storage));

    return sizes;
}

std::vector<std::tuple<uint32_t, uint32_t>> SoftmaxOpL1Usage::SoftmaxOpL1Usage::get_tensor_l1_allocations_per_core()
    const {
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
