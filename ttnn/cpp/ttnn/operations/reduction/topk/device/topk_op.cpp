// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_op.hpp"
#include "topk_program_factory.hpp"

using namespace tt::tt_metal;

namespace topk_utils {

static inline uint32_t largest_power_of_two(std::uint32_t x) { return x == 0 ? 0 : (1 << (31 - __builtin_clz(x))); }

static inline bool verify_multi_core_cost(
    const std::vector<Tensor>& input_tensors,
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const CoreRangeSet& core_range_set) {
    auto device = input_tensors.at(0).device();
    tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).dtype());
    tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;

    uint32_t value_tile_size = tile_size(value_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    const auto core_range = core_range_set.ranges().at(0);
    const auto max_cores = core_range.end_coord.y - core_range.start_coord.y - 1;
    uint32_t start_split_size = width / largest_power_of_two(max_cores);
    for (uint32_t split_size = start_split_size; split_size <= max_dim; split_size *= 2) {
        uint32_t rem = width % split_size;
        uint32_t num_cores = width / split_size + (rem > 0);
        uint32_t memory_cost_gather =
            2 * num_cores * (value_tile_size + index_tile_size);  // gathering one index and one value tile from each
                                                                  // local core, allocating two CBs for each
        uint32_t memory_cost_local =
            (split_size / tt::constants::TILE_WIDTH) *
            (value_tile_size + index_tile_size);  // we divide the width into split_size chunks and each chunk, as well
                                                  // as a matching set of indices, is processed by a core
        if (num_cores <= max_cores &&
            (memory_cost_gather + (memory_cost_local * num_cores)) < (device->l1_size_per_core() * num_cores) &&
            num_cores > 1 && split_size >= min_dim) {
            return true;
        }
    }
    return false;
}

static inline bool verify_single_core_cost(const std::vector<Tensor>& input_tensors, uint32_t k) {
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;
    uint32_t Ktiles = tt::div_up(k, tt::constants::TILE_WIDTH);
    uint32_t input_cb_tile_count = cb_in_units;
    uint32_t transposed_cb_tile_count = 4;
    uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // intermediate output
    uint32_t output_cb_tile_count = Ktiles;

    auto device = input_tensors.at(0).device();
    tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).dtype());
    tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;

    uint32_t value_tile_size = tile_size(value_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    uint32_t memory_cost_local =
        (input_cb_tile_count + transposed_cb_tile_count + result_prep_cb_tile_count + output_cb_tile_count) *
        (value_tile_size + index_tile_size);  // we divide the width into split_size chunks and each chunk, as well
                                              // as a matching set of indices, is processed by a core
    return memory_cost_local < device->l1_size_per_core();
}

constexpr uint32_t multi_core_min_width = 8192;
constexpr uint32_t min_dim_per_core = 64;
}  // namespace topk_utils
namespace ttnn::operations::reduction {

void TopK::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    auto input_shape = input_tensors.at(0).padded_shape();
    TT_FATAL(input_shape.rank() == 4, "Input shape must be 4D, got {}", input_shape.rank());

    TT_FATAL(
        input_shape[-1] >= topk_utils::min_dim_per_core,
        "Input shape inner dim {} must be a multiple of 64, pad with +/-infinity if necessary",
        input_shape[-1]);
    TT_FATAL(
        (input_shape[0] * input_shape[1] * input_shape[2]) % 32 == 0,
        "Input height (combined input_shape[0-3]) {} must be a multiple of 32",
        input_shape[0] * input_shape[1] * input_shape[2]);

    TT_FATAL(this->output_mem_config.is_sharded() == false, "Sharded implementation not supported yet");
    TT_FATAL(input_tensors.at(0).layout() == Layout::TILE, "The input must be in tiled format");

    bool can_run = false;

    if (input_shape[dim] >= topk_utils::multi_core_min_width) {  // multicore implementation
        can_run = topk_utils::verify_multi_core_cost(
            input_tensors,
            input_shape[this->dim],
            topk_utils::min_dim_per_core,
            input_shape[this->dim] / 2,
            this->k,
            this->sub_core_grids);

        TT_FATAL(
            this->sub_core_grids.ranges().size() == 1,
            "Only one core range is supported right now, got {}",
            this->sub_core_grids.ranges().size());

        if (!can_run) {  // can we default to new topk implementation on single core
            can_run = topk_utils::verify_single_core_cost(input_tensors, this->k);
        }
    } else {
        can_run = topk_utils::verify_single_core_cost(input_tensors, this->k);
    }
    TT_FATAL(can_run, "Not enough cores or cache size available to run topk operation");
}

std::vector<TensorSpec> TopK::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 2) {
        if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
            return {output_tensors[0]->tensor_spec(), output_tensors[1]->tensor_spec()};
        }
    }
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = input_tensors.at(0).logical_shape();
    output_shape[-1] = this->k;

    auto values_spec =
        TensorSpec(output_shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), output_mem_config));
    auto index_spec =
        TensorSpec(output_shape, TensorLayout(DataType::UINT16, PageConfig(Layout::TILE), output_mem_config));
    return {values_spec, index_spec};
}

std::vector<Tensor> TopK::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 2) {
        if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
            return {output_tensors[0].value(), output_tensors[1].value()};
        }
    }
    auto output_specs = compute_output_specs(input_tensors, output_tensors);
    const auto& input_tensor = input_tensors.at(0);
    return {
        create_device_tensor(output_specs[0], input_tensor.device()),
        create_device_tensor(output_specs[1], input_tensor.device()),
    };
}

operation::ProgramWithCallbacks TopK::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    bool multicore_supported = true;
    multicore_supported &= (input_tensor.padded_shape()[dim] >= topk_utils::multi_core_min_width);

    auto input_shape = input_tensors.at(0).padded_shape();
    auto device = input_tensors.at(0).device();

    tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).dtype());
    tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;
    uint32_t value_tile_size = tile_size(value_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    // There is a L1 cache issue for multicore implementation (previous version) because footprint depends on input
    // tensor size. New implementation takes up less L1 cache size because footprint is based on K size, and not input
    // tensor size, so it can handle larger input tensor sizes.
    // TODO: implement new topk implementation for multicore
    multicore_supported &= topk_utils::verify_multi_core_cost(
        input_tensors,
        input_shape[this->dim],
        topk_utils::min_dim_per_core,
        input_shape[this->dim] / 2,
        this->k,
        this->sub_core_grids);

    multicore_supported &= (this->k <= 64);  // old implementation cannot handle k>64

    if (multicore_supported) {
        return detail::topk_multicore_interleaved(
            input_tensor,
            this->k,
            this->dim,
            this->largest,
            this->sorted,
            this->sub_core_grids,
            output_tensors.at(0),
            output_tensors.at(1));
    }

    return detail::topk_single_core_interleaved(
        input_tensor,
        this->k,
        this->dim,
        this->largest,
        this->sorted,
        this->sub_core_grids,
        output_tensors.at(0),
        output_tensors.at(1));
}

}  // namespace ttnn::operations::reduction
