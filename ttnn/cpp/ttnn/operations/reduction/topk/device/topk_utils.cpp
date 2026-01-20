// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

namespace ttnn::prim {

uint32_t largest_power_of_two(uint32_t x) { return x == 0 ? 0 : (1U << (31 - __builtin_clz(x))); }

std::optional<TopKCoreConfig> find_topk_core_config(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size) {
    const auto max_cores =
        (core_range.end_coord.y - core_range.start_coord.y - 1) * (core_range.end_coord.x - core_range.start_coord.x);
    uint32_t start_split_size =
        static_cast<uint32_t>(width / tt::constants::TILE_WIDTH / largest_power_of_two(max_cores)) *
        tt::constants::TILE_WIDTH;
    for (uint32_t split_size = start_split_size; split_size <= max_dim; split_size *= 2) {
        uint32_t rem = width % split_size;
        uint32_t num_cores = (width / split_size) + (rem > 0);
        uint32_t memory_cost_gather = 2 * num_cores * (value_tile_size + index_tile_size);
        uint32_t memory_cost_local = (split_size / tt::constants::TILE_WIDTH) * (value_tile_size + index_tile_size);
        uint32_t max_x = core_range.end_coord.x - core_range.start_coord.x;
        uint32_t max_y = core_range.end_coord.y - core_range.start_coord.y - 1;
        uint32_t max_cores_available = max_x * max_y;
        if (num_cores > max_cores_available) {
            continue;
        }
        bool contiguous_cores_available = false;
        uint32_t selected_x = 0;
        uint32_t selected_y = 0;
        for (uint32_t y = max_y; y > 0; y--) {
            for (uint32_t x = max_x; x > 0; x--) {
                if (x * y == num_cores) {
                    selected_x = x;
                    selected_y = y;
                    contiguous_cores_available = true;
                    break;
                }
            }
        }
        if (num_cores <= max_cores && memory_cost_gather + (memory_cost_local * num_cores) < (l1_size * num_cores) &&
            num_cores > 1 && split_size >= min_dim && contiguous_cores_available && rem == 0) {
            TopKCoreConfig config;
            config.num_cores = static_cast<uint16_t>(num_cores);
            config.split_size = static_cast<uint16_t>(split_size);
            config.rem = static_cast<uint16_t>(rem);
            config.final_input_size =
                static_cast<uint16_t>(num_cores * std::max(k, static_cast<uint32_t>(tt::constants::TILE_WIDTH)));
            config.selected_x = static_cast<uint16_t>(selected_x);
            config.selected_y = static_cast<uint16_t>(selected_y);
            return std::make_optional(config);
        }
    }
    return std::nullopt;
}

bool verify_multi_core_cost(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size) {
    auto config =
        find_topk_core_config(width, min_dim, max_dim, k, core_range, l1_size, value_tile_size, index_tile_size);
    return config.has_value();
}

bool verify_single_core_cost(const ttnn::Tensor& input_tensor, uint32_t k, bool uint16_output) {
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;
    uint32_t Ktiles = tt::div_up(k, tt::constants::TILE_WIDTH);
    uint32_t input_cb_tile_count = cb_in_units;
    uint32_t transposed_cb_tile_count = 4;
    uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // intermediate output
    uint32_t output_cb_tile_count = Ktiles;
    auto* device = input_tensor.device();
    tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat index_cb_data_format = uint16_output ? tt::DataFormat::UInt16 : tt::DataFormat::UInt32;
    uint32_t value_tile_size = tt::tile_size(value_cb_data_format);
    uint32_t index_tile_size = tt::tile_size(index_cb_data_format);
    uint32_t memory_cost_local =
        (input_cb_tile_count + transposed_cb_tile_count + result_prep_cb_tile_count + output_cb_tile_count) *
        (value_tile_size + index_tile_size);
    return memory_cost_local < device->l1_size_per_core();
}

}  // namespace ttnn::prim
