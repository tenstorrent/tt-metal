// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_op.hpp"
#include "moe_program_factory.hpp"

namespace moe_utils {

// static inline bool verify_available_cores(uint16_t width, uint16_t min_dim, uint16_t max_dim, CoreCoord grid, uint16_t k, const uint32_t value_tile_size, const uint32_t index_tile_size) {
//     const auto max_cores = grid.y - 1; // reserve one core for the gather - switch to grid.x as it allows for more cores and allow spillover to next row
//     for (uint16_t split_size = max_dim; split_size >= min_dim; split_size/=2) {
//         uint16_t rem = width % split_size;
//         uint16_t num_cores = width / split_size + (rem > 0);
//         uint32_t memory_cost_gather = 2*num_cores * (value_tile_size + index_tile_size); // gathering one index and one value tile from each local core, allocating two CBs for each
//         uint32_t memory_cost_local = (split_size / TILE_WIDTH) * (value_tile_size + index_tile_size); // we divide the width into split_size chunks and each chunk, as well as a matching set of indices, is processed by a core
//         if (num_cores <= max_cores && (memory_cost_gather + memory_cost_local) < L1_SIZE && num_cores > 1) {
//             return true;
//         }
//     }
//     return false;
// }

constexpr uint32_t multi_core_min_width = 8192;
}
namespace ttnn::operations::reduction {

void Moe::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    auto input_shape = input_tensors.at(0).get_legacy_shape();
    TT_FATAL(input_shape.rank() == 4, fmt::format("Input shape must be 4D, got {}", input_shape.rank()));
    TT_FATAL(this->k == 32, fmt::format("K must be equal to 32, pad with -infinity if necessary to get 32, got {}", this->k));

    TT_FATAL(input_shape[-1] >= 64, fmt::format("Input shape inner dim {} must be a multiple of 64, pad with -infinity if necessary", input_shape[-1]));
    TT_FATAL((input_shape[-1] & (input_shape[-1] - 1)) == 0, fmt::format("Input shape inner dim {} must be a power of 2, pad with -infinity if necessary", input_shape[-1]));
    TT_FATAL((input_shape[0] * input_shape[1] * input_shape[2]) % 32 == 0, fmt::format("Input height (combined input_shape[0-3]) {} must be a multiple of 32", input_shape[0] * input_shape[1] * input_shape[2]));

    TT_FATAL(this->output_mem_config.is_sharded() == false, "Sharded implementation not supported yet");
    TT_FATAL(input_tensors.at(0).get_layout() == Layout::TILE, "The input must be in tiled format");

    auto topk_shape = input_tensors.at(1).get_legacy_shape();
    auto expert_shape = input_tensors.at(2).get_legacy_shape();

    TT_FATAL(topk_shape[-1] == this->k, fmt::format("Topk shape inner dim must be equal to k, got {}", topk_shape[-1]));
    TT_FATAL(expert_shape[-1] == this->k, fmt::format("Expert shape inner dim must be equal to k, got {}", expert_shape[-1]));
    TT_FATAL(topk_shape[-2] == 32, fmt::format("Topk shape inner dim must be equal to 1, got {}", topk_shape[-2]));
    TT_FATAL(expert_shape[-2] == 32, fmt::format("Expert shape inner dim must be equal to 1, got {}", expert_shape[-2]));

    // if (input_shape[dim] >=  moe_utils::multi_core_min_width) { // multicore implementation
    //     auto device = input_tensors.at(0).device();

    //     tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).get_dtype());
    //     tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;

    //     uint32_t value_tile_size = tile_size(value_cb_data_format);
    //     uint32_t index_tile_size = tile_size(index_cb_data_format);
    //     TT_FATAL(moe_utils::verify_available_cores(input_shape[this->dim], 64, input_shape[this->dim]/2, device->compute_with_storage_grid_size(),
    //                         this->k, value_tile_size, index_tile_size), "Not enough cores available to run moe operation");
    // }
}

std::vector<tt::tt_metal::Shape> Moe::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    return {{input_shape[0], input_shape[1], input_shape[2], 32}};
}

std::vector<Tensor> Moe::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (output_tensors.size() == 1) {
        if (output_tensors.at(0).has_value()) {
            return {output_tensors[0].value()};
        }
    }
    const auto& input_tensor = input_tensors.at(0);
    const auto shapes = compute_output_shapes(input_tensors);
    auto out_tensor = create_device_tensor(shapes[0], input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), this->output_mem_config);
    return {out_tensor};
}

operation::ProgramWithCallbacks Moe::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    return detail::moe_single_core_interleaved(input_tensors.at(0), input_tensors.at(1), input_tensors.at(2), this->k, output_tensors.at(0));
    // if (input_tensor.get_legacy_shape()[dim] < moe_utils::multi_core_min_width) {
    //     return detail::moe_single_core_interleaved(input_tensor, this->k, this->dim, output_tensors.at(0), output_tensors.at(1));
    // } else {
    //     return detail::moe_multicore_interleaved(input_tensor, this->k, this->dim, output_tensors.at(0), output_tensors.at(1));
    // }
}

}  // namespace ttnn::operations::reduction
