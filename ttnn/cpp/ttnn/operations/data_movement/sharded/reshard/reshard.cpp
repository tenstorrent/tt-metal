// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "device/reshard_op.hpp"
#include "reshard.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

std::vector<uint32_t> get_runtime_args_for_given_ranges(
    const std::vector<uint32_t>& physical_core_coords,
    const std::vector<detail::PageStride>& page_stride_vector,
    const uint32_t output_page_offset,
    const uint32_t& input_addr,
    const uint32_t starting_range,
    const uint32_t ending_range,
    const detail::ReshardStridesInRange reshard_strides_in_range) {
    std::vector<uint32_t> runtime_args = physical_core_coords;
    runtime_args.push_back(input_addr);
    runtime_args.push_back(0);
    runtime_args.push_back(ending_range - starting_range);
    runtime_args.push_back(output_page_offset);
    uint32_t num_output_pages = 0;

    for (uint32_t range_id = starting_range; range_id < ending_range; range_id++) {
        detail::PageStride ps = page_stride_vector[range_id];
        uint32_t num_strides;
        uint32_t start_core_x;
        uint32_t start_core_y;
        uint32_t start_data;
        if (reshard_strides_in_range == detail::ReshardStridesInRange::ALL_STRIDES) {
            num_strides = ps.num_strides;
            start_core_x = ps.start_core.x;
            start_core_y = ps.start_core.y;
            start_data = ps.start_data;
        } else {
            if (reshard_strides_in_range == detail::ReshardStridesInRange::FIRST_HALF) {
                num_strides = ps.num_strides / 2;
                start_core_x = ps.start_core.x;
                start_core_y = ps.start_core.y;
                start_data = ps.start_data;
            } else {
                uint32_t strides_in_first_half = ps.num_strides / 2;
                num_strides = ps.num_strides - (strides_in_first_half);
                start_core_x = ps.start_core.x + (strides_in_first_half * ps.stride.core.x);
                start_core_y = ps.start_core.y + (strides_in_first_half * ps.stride.core.y);
                start_data = ps.start_data + (strides_in_first_half * ps.start_data);
            }
        }
        if (num_strides > 0) {
            uint32_t core_start_stride =
                (start_core_x << 24) | (start_core_y << 16) | (ps.stride.core.x << 8) | ps.stride.core.y;
            runtime_args.push_back((uint32_t)core_start_stride);  // start_x
            uint32_t stride_data_start = (ps.stride.data << 16) | (start_data);
            runtime_args.push_back((uint32_t)stride_data_start);  // stride_data
            uint32_t stride_size_num_strides = (ps.stride_size << 16) | (num_strides << 8) | ((uint32_t)ps.skip);
            runtime_args.push_back((uint32_t)stride_size_num_strides);  // stride_size
            num_output_pages += ps.stride_size * num_strides;
        }
    }
    runtime_args[physical_core_coords.size() + 1] = num_output_pages;
    return runtime_args;
}

std::unordered_map<CoreCoord, std::vector<detail::PageStride>> create_map_for_reshard(
    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page,
    Buffer* input_buffer,
    Buffer* output_buffer) {
    std::unordered_map<CoreCoord, std::vector<detail::PageStride>> ret_map;
    auto output_cores = output_buffer->get_buffer_page_mapping()->all_cores;
    ret_map.reserve(output_cores.size());

    auto device = input_buffer->device();
    auto full_grid = device->compute_with_storage_grid_size();
    uint32_t output_core_id = 0;
    for (auto output_core : output_cores) {
        ret_map.try_emplace(output_core, std::vector<detail::PageStride>{});

        const auto& input_cores_with_pages = output_core_to_vector_input_core_page[output_core_id];
        auto it = input_cores_with_pages.begin();
        const auto end = input_cores_with_pages.end();

        while (it != end) {
            // hit padding, will see how many consecutive pages has padding to make a padded range
            if (!it->has_value()) {
                auto consecutive_it = it + 1;
                auto last_it_consec = it;
                while (consecutive_it != end) {
                    if (consecutive_it->has_value()) {
                        break;
                    }
                    last_it_consec = consecutive_it;
                    consecutive_it = consecutive_it + 1;
                }
                uint32_t stride_size = std::distance(it, last_it_consec) + 1;
                ret_map[output_core].push_back(detail::PageStride{
                    .start_core = output_core,
                    .start_data = 0,
                    .stride_size = stride_size,
                    .stride = detail::Stride{.core = {0, 0}, .data = 0},
                    .num_strides = 1,
                    .skip = true});
                it += stride_size;
            } else {
                const auto start_core = it->value().first;
                detail::Stride stride = detail::Stride{.core = {0, 0}, .data = 0};
                if ((it + 1) == end) {
                    ret_map[output_core].push_back(detail::PageStride{
                        .start_core = start_core,
                        .start_data = it->value().second,
                        .stride_size = 1,
                        .stride = stride,
                        .num_strides = 1,
                        .skip = false});
                    it = end;
                } else {
                    // first get a single stride, go through the number of consecutive pages in the same core
                    auto consecutive_it = it + 1;
                    auto last_it_consec = it;
                    while (consecutive_it != end and consecutive_it->has_value()) {
                        auto next_input_page = *(consecutive_it);
                        auto curr_input_page = *(last_it_consec);
                        // diff core , not consecutive
                        if (curr_input_page.value().first != next_input_page.value().first) {
                            break;
                        }
                        // not consecutive
                        else if ((curr_input_page.value().second + 1) != next_input_page.value().second) {
                            break;
                        }
                        // next page is padding
                        last_it_consec = consecutive_it;
                        consecutive_it = consecutive_it + 1;
                    }
                    uint32_t stride_size = std::distance(it, last_it_consec) + 1;
                    auto stride_it = it + stride_size;
                    auto last_it_stride = it;

                    // TT_ASSERT((stride_it == end) or stride_it->has_value());
                    TT_ASSERT(last_it_stride->has_value());
                    // if stride_range is within same core
                    // the jump in data is end of curr - end last stride
                    // if stride range is in diff core
                    // jump in data is curr - beginning of last stride
                    uint32_t data_stride;
                    if ((stride_it != end) and (stride_it != it) and stride_it->has_value()) {
                        // data stride within core
                        if (stride_it->has_value() and stride_it->value().first == last_it_stride->value().first and
                            (stride_it->value().second > last_it_stride->value().second)) {
                            auto next_input_page = *(stride_it);
                            auto prev_input_page = *(last_it_stride);
                            TT_ASSERT(prev_input_page.has_value());
                            TT_ASSERT(next_input_page.has_value());
                            data_stride = next_input_page.value().second - prev_input_page.value().second - stride_size;
                            stride = detail::Stride{.core = {0, 0}, .data = data_stride};
                        }
                        // strided core but same data
                        // currently only handling increasing cores within same stride
                        // TODO : negative strides for cores
                        else if (
                            stride_it->has_value() and (stride_it->value().first != last_it_stride->value().first) and
                            (stride_it->value().first.x >= it->value().first.x and
                             stride_it->value().first.y >= it->value().first.y) and
                            (stride_it->value().second == it->value().second)) {
                            auto next_input_page = *(stride_it);
                            auto prev_input_page = *it;
                            TT_ASSERT(prev_input_page.has_value());
                            TT_ASSERT(next_input_page.has_value());
                            data_stride = 0;
                            stride = detail::Stride{
                                .core =
                                    {next_input_page.value().first.x - prev_input_page.value().first.x,
                                     next_input_page.value().first.y - prev_input_page.value().first.y},
                                .data = data_stride};
                        }
                        // diff data and diff core, not handled yet
                        else {
                            TT_ASSERT(it->has_value());
                            ret_map[output_core].push_back(detail::PageStride{
                                .start_core = start_core,
                                .start_data = it->value().second,
                                .stride_size = stride_size,
                                .stride = stride,
                                .num_strides = 1,
                                .skip = false});
                            it = stride_it;
                            continue;
                        }
                        // TODO add stride of data and core
                    }
                    // only single stride
                    else {
                        data_stride = 0;
                    }

                    TT_ASSERT(stride.core.x < full_grid.x and stride.core.y < full_grid.y);
                    TT_ASSERT(data_stride < output_buffer->num_pages());
                    uint32_t num_strides = 1;
                    while (stride_it != end and stride_it->has_value()) {
                        bool stride_not_complete = false;
                        auto stride_it_inner = stride_it + 1;
                        auto last_it_stride_inner = stride_it;
                        for (uint32_t i = 0; i < stride_size - 1; i++) {
                            auto next_input_page = *(stride_it_inner);
                            auto curr_input_page = *(last_it_stride_inner);
                            TT_ASSERT(curr_input_page.has_value());
                            int increment = 1;
                            if (!(next_input_page.has_value()) or
                                (next_input_page.value().first != curr_input_page.value().first) or
                                ((int)next_input_page.value().second !=
                                 (int)(curr_input_page.value().second) + (int)increment)) {
                                stride_not_complete = true;
                                break;
                            }
                            last_it_stride_inner = stride_it_inner;
                            stride_it_inner = stride_it_inner + 1;
                        }
                        if (stride_not_complete) {
                            break;
                        }
                        num_strides++;
                        last_it_stride = stride_it_inner - 1;
                        stride_it = stride_it_inner;
                        if (stride_it == end or !stride_it->has_value()) {
                            break;
                        }
                        auto next_input_page = *(stride_it);
                        auto curr_input_page = *(last_it_stride);
                        // TT_ASSERT(curr_input_page.has_value());
                        if (!curr_input_page.has_value() or !next_input_page.has_value() or
                            (next_input_page.value().first.x - curr_input_page.value().first.x != stride.core.x) or
                            (next_input_page.value().first.y - curr_input_page.value().first.y != stride.core.y) or
                            (abs((int)next_input_page.value().second - (int)curr_input_page.value().second) !=
                             (int)stride.data)) {
                            break;
                        }
                    }
                    TT_ASSERT(it->has_value());
                    ret_map[output_core].push_back(detail::PageStride{
                        .start_core = start_core,
                        .start_data = it->value().second,
                        .stride_size = stride_size,
                        .stride = stride,
                        .num_strides = num_strides,
                        .skip = false});
                    it = stride_it;
                }
            }
        }
        output_core_id++;
    }
    return ret_map;
}

std::unordered_map<CoreCoord, std::vector<detail::PageStride>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer) {
    const auto& output_buffer_page_mapping = *output_buffer->get_buffer_page_mapping();
    const auto& input_buffer_page_mapping = *input_buffer->get_buffer_page_mapping();

    std::vector<std::pair<CoreCoord, uint32_t>> host_page_to_input_core_mapping(input_buffer->num_pages());
    for (auto mapped_page : input_buffer_page_mapping) {
        auto core = input_buffer_page_mapping.all_cores[mapped_page.core_id];
        host_page_to_input_core_mapping[mapped_page.host_page] = {core, mapped_page.device_page};
    }

    auto output_cores = output_buffer_page_mapping.all_cores;

    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page(
        output_cores.size());

    for (auto mapped_page : output_buffer_page_mapping) {
        auto& cur_output_core_to_vector_input_core_page = output_core_to_vector_input_core_page[mapped_page.core_id];
        auto [input_core, input_core_page] = host_page_to_input_core_mapping[mapped_page.host_page];
        if (cur_output_core_to_vector_input_core_page.size() <= mapped_page.device_page) {
            cur_output_core_to_vector_input_core_page.resize(mapped_page.device_page + 1);
        }
        cur_output_core_to_vector_input_core_page[mapped_page.device_page] = {input_core, input_core_page};
    }
    auto ret_map = create_map_for_reshard(output_core_to_vector_input_core_page, input_buffer, output_buffer);
    return ret_map;
}

std::unordered_map<CoreCoord, std::vector<detail::PageStride>> get_core_page_ranges_diff_width(
    Buffer* input_buffer, Buffer* output_buffer, const Tensor& input) {
    const auto& output_buffer_page_mapping = *output_buffer->get_buffer_page_mapping();
    const auto& input_buffer_page_mapping = *input_buffer->get_buffer_page_mapping();
    uint32_t num_rows = 1;
    for (uint32_t i = 0; i < input.logical_shape().rank() - 1; i++) {
        num_rows *= input.logical_shape()[i];
    }
    // Find GCD of page sizes to use as the new base page size
    uint32_t input_page_size = input_buffer->page_size();
    uint32_t output_page_size = output_buffer->page_size();
    uint32_t base_page_size = std::gcd(input_page_size, output_page_size);

    // Calculate how many base pages make up an input/output page
    uint32_t input_pages_per_original = input_page_size / base_page_size;
    uint32_t output_pages_per_original = output_page_size / base_page_size;

    auto input_width = input_buffer->shard_spec().shape()[1];
    auto output_width = output_buffer->shard_spec().shape()[1];
    auto total_width = input.logical_shape()[-1];

    uint32_t total_page_number =
        std::ceil((float)(input.logical_shape()[-1] * input.element_size()) / (float)base_page_size);
    uint32_t num_input_pages_per_row = input_pages_per_original * std::ceil((float)total_width / (float)input_width);
    uint32_t num_output_pages_per_row = output_pages_per_original * std::ceil((float)total_width / (float)output_width);

    std::vector<std::pair<CoreCoord, uint32_t>> host_page_to_input_core_mapping(total_page_number * num_rows);

    // data structure to account for padded base pages in the mapping
    std::unordered_map<uint32_t, uint32_t> invalid_mapping_input;
    std::unordered_map<uint32_t, uint32_t> invalid_mapping_output;
    uint32_t num_invalid_pages_input = 0;
    uint32_t num_invalid_pages_output = 0;
    for (uint32_t i = 1; i <= num_rows; i++) {
        invalid_mapping_input[total_page_number * i] = 0;
        invalid_mapping_output[total_page_number * i] = 0;
    }

    // find input invalid base pages if applicable
    for (auto mapped_page : input_buffer_page_mapping) {
        auto core = input_buffer_page_mapping.all_cores[mapped_page.core_id];
        CoreCoord shard_grid = input_buffer->shard_spec().grid().ranges()[0].grid_size();
        bool is_last_in_row = (core.x == shard_grid.x - 1);
        if (input_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            is_last_in_row = (core.y == shard_grid.y - 1);
        }
        uint32_t base_start_page = mapped_page.host_page * input_pages_per_original;
        uint32_t device_base_start = mapped_page.device_page * input_pages_per_original;
        uint32_t valid_pages = input_pages_per_original;
        if (is_last_in_row) {
            uint32_t next_total =
                ((base_start_page + num_input_pages_per_row) / num_input_pages_per_row) * total_page_number;
            next_total = std::max(next_total, total_page_number);
            valid_pages = std::min(next_total - base_start_page, input_pages_per_original);
        }
        if (input_pages_per_original - valid_pages > 0) {
            num_invalid_pages_input = input_pages_per_original - valid_pages;
            break;
        }
    }

    // find output invalid base pages if applicable
    for (auto mapped_page : output_buffer_page_mapping) {
        auto core = output_buffer_page_mapping.all_cores[mapped_page.core_id];
        CoreCoord shard_grid = output_buffer->shard_spec().grid().ranges()[0].grid_size();
        bool is_last_in_row = (core.x == shard_grid.x - 1);
        if (output_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            is_last_in_row = (core.y == shard_grid.y - 1);
        }
        uint32_t base_start_page = mapped_page.host_page * output_pages_per_original;
        uint32_t device_base_start = mapped_page.device_page * output_pages_per_original;
        uint32_t valid_pages = output_pages_per_original;
        if (is_last_in_row) {
            uint32_t next_total =
                ((base_start_page + num_output_pages_per_row) / num_output_pages_per_row) * total_page_number;
            valid_pages = std::min(next_total - base_start_page, output_pages_per_original);
        }
        if (output_pages_per_original - valid_pages > 0) {
            num_invalid_pages_output = output_pages_per_original - valid_pages;
            break;
        }
    }

    for (uint32_t i = 1; i <= num_rows; i++) {
        invalid_mapping_input[total_page_number * i] = (i - 1) * num_invalid_pages_input;
        invalid_mapping_output[total_page_number * i] = (i - 1) * num_invalid_pages_output;
    }

    // Create mapping of input base host pages to their cores
    for (auto mapped_page : input_buffer_page_mapping) {
        auto core = input_buffer_page_mapping.all_cores[mapped_page.core_id];
        uint32_t base_start_page = mapped_page.host_page * input_pages_per_original;
        uint32_t device_base_start = mapped_page.device_page * input_pages_per_original;
        uint32_t next_total =
            ((base_start_page + num_input_pages_per_row) / num_input_pages_per_row) * total_page_number;
        next_total = std::max(next_total, total_page_number);
        base_start_page = base_start_page - invalid_mapping_input[next_total];
        uint32_t valid_pages = std::min(next_total - base_start_page, input_pages_per_original);
        for (uint32_t i = 0; i < valid_pages; i++) {
            host_page_to_input_core_mapping[base_start_page + i] = {core, device_base_start + i};
        }
    }

    // Create similar mapping for output pages to their cores
    std::vector<std::pair<CoreCoord, uint32_t>> host_page_to_output_core_mapping(total_page_number * num_rows);

    for (auto mapped_page : output_buffer_page_mapping) {
        auto core = output_buffer_page_mapping.all_cores[mapped_page.core_id];
        uint32_t base_start_page = mapped_page.host_page * output_pages_per_original;
        uint32_t device_base_start = mapped_page.device_page * output_pages_per_original;

        uint32_t next_total =
            ((base_start_page + num_output_pages_per_row) / num_output_pages_per_row) * total_page_number;
        next_total = std::max(next_total, total_page_number);
        base_start_page = base_start_page - invalid_mapping_output[next_total];
        uint32_t valid_pages = std::min(next_total - base_start_page, output_pages_per_original);
        for (uint32_t i = 0; i < valid_pages; i++) {
            host_page_to_output_core_mapping[base_start_page + i] = {core, device_base_start + i};
        }
    }
    // Create final mapping of output cores to input pages they need
    auto output_cores = output_buffer_page_mapping.all_cores;
    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page(
        output_cores.size());

    for (uint32_t core_id = 0; core_id < output_cores.size(); core_id++) {
        auto& cur_output_core_pages = output_core_to_vector_input_core_page[core_id];

        // Find all host pages that map to this output core
        for (uint32_t host_page = 0; host_page < host_page_to_output_core_mapping.size(); host_page++) {
            if (host_page_to_output_core_mapping[host_page].first == output_cores[core_id]) {
                // This host page belongs to current output core
                // Get corresponding input core and page
                auto input_mapping = host_page_to_input_core_mapping[host_page];

                // Add to vector if needed
                uint32_t device_page = host_page_to_output_core_mapping[host_page].second;
                if (cur_output_core_pages.size() <= device_page) {
                    cur_output_core_pages.resize(device_page + 1);
                }
                cur_output_core_pages[device_page] = input_mapping;
            }
        }
    }

    auto ret_map = create_map_for_reshard(output_core_to_vector_input_core_page, input_buffer, output_buffer);
    return ret_map;
}

Tensor construct_per_core_host_tensor(const std::unordered_map<CoreCoord, std::vector<uint32_t>>& core_to_data) {
    // Find max vector size to determine tensor width
    size_t max_width = 0;
    for (const auto& [core, data] : core_to_data) {
        max_width = std::max(max_width, data.size());
    }
    // Create shape based on number of cores and max width
    ttnn::Shape tensor_shape({static_cast<uint32_t>(core_to_data.size()), static_cast<uint32_t>(max_width)});
    // Sort cores to ensure consistent ordering
    std::vector<CoreCoord> ordered_cores;
    ordered_cores.reserve(core_to_data.size());
    for (const auto& [core, _] : core_to_data) {
        ordered_cores.push_back(core);
    }
    std::sort(ordered_cores.begin(), ordered_cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return (a.y < b.y) || (a.y == b.y && a.x < b.x);
    });
    // Flatten data from all cores into single vector with padding
    std::vector<uint32_t> flattened_data;
    flattened_data.reserve(core_to_data.size() * max_width);
    for (const auto& core : ordered_cores) {
        const auto& data = core_to_data.at(core);
        flattened_data.insert(flattened_data.end(), data.begin(), data.end());
        // Add padding if needed
        if (data.size() < max_width) {
            flattened_data.insert(flattened_data.end(), max_width - data.size(), 0);
        }
    }
    // Create host buffer and tensor
    auto config_buffer = tt::tt_metal::HostBuffer(std::move(flattened_data));
    return Tensor(std::move(config_buffer), tensor_shape, DataType::UINT32, Layout::ROW_MAJOR);
}
Tensor move_per_core_config_to_device(
    const Tensor& host_tensor, const CoreRangeSet& grid, distributed::MeshDevice* device) {
    // Create shard spec for the config tensor
    // Each core gets a row of the tensor
    const std::array<uint32_t, 2> shard_shape = {1, host_tensor.logical_shape()[1]};
    auto shard_spec = tt::tt_metal::ShardSpec(grid, shard_shape, ShardOrientation::ROW_MAJOR);
    // Create memory config for device tensor
    auto mem_config = MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec);
    return host_tensor.to_device(device, mem_config);
}

ttnn::Tensor ReshardOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TensorSpec output_tensor_spec = optional_output_tensor.has_value() ? optional_output_tensor->tensor_spec()
                                                                       : TensorSpec(
                                                                             input_tensor.logical_shape(),
                                                                             TensorLayout::fromPaddedShape(
                                                                                 input_tensor.dtype(),
                                                                                 input_tensor.layout(),
                                                                                 memory_config,
                                                                                 input_tensor.logical_shape(),
                                                                                 input_tensor.padded_shape()));

    auto device = input_tensor.device();
    auto output_tensor = create_device_tensor(output_tensor_spec, device);

    bool not_generic =
        (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
         memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED);
    not_generic = not_generic || (input_tensor.layout() == Layout::ROW_MAJOR &&
                                  input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                                  memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);
    std::unordered_map<CoreCoord, std::vector<uint32_t>> rt_config_map_0;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> rt_config_map_1;
    std::vector<Tensor> inputs;
    inputs.push_back(input_tensor);
    if (!not_generic && output_tensor.shard_spec().has_value()) {
        std::unordered_map<CoreCoord, std::vector<detail::PageStride>> output_core_to_page_range_pair;
        if (input_tensor.buffer()->page_size() != output_tensor.buffer()->page_size()) {
            output_core_to_page_range_pair =
                get_core_page_ranges_diff_width(input_tensor.buffer(), output_tensor.buffer(), input_tensor);
        } else {
            output_core_to_page_range_pair = get_core_page_ranges(input_tensor.buffer(), output_tensor.buffer());
        }
        const auto& input = input_tensor;
        const auto& output = output_tensor;
        auto output_shard_spec = output.shard_spec().value();
        auto all_cores = output_shard_spec.grid;
        auto grid = input.buffer()->buffer_type() == BufferType::DRAM ? device->dram_grid_size()
                                                                      : device->compute_with_storage_grid_size();
        auto input_core_type = input.buffer()->core_type();
        auto cores =
            corerange_to_cores(all_cores, std::nullopt, output_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

        uint32_t total_size, page_size, unit_size;
        auto output_shard_shape = output_shard_spec.shape;
        auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

        if (input.layout() == Layout::TILE) {
            page_size = tt::tt_metal::detail::TileSize(data_format);
            unit_size = page_size;
            total_size = output_shard_spec.numel() / tt::constants::TILE_HW * unit_size;
        } else {
            // For ROW_MAJOR, use base page size from GCD calculation
            uint32_t input_page_size = input.buffer()->page_size();
            uint32_t output_page_size = output.buffer()->page_size();
            uint32_t base_page_size = std::gcd(input_page_size, output_page_size);

            unit_size = base_page_size;
            page_size = base_page_size;
            total_size = output_shard_shape[0] * output_shard_shape[1] * output.element_size();
        }
        std::vector<uint32_t> physical_core_coords;
        physical_core_coords.reserve(grid.x * grid.y);
        for (uint32_t i = 0; i < grid.x; i++) {
            auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(i, 0), input_core_type);
            physical_core_coords.push_back(physical_input_core.x);
        }
        for (uint32_t i = 0; i < grid.y; i++) {
            auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(0, i), input_core_type);
            physical_core_coords.push_back(physical_input_core.y);
        }
        for (const auto& core : cores) {
            const auto& page_stride_vector = output_core_to_page_range_pair.at(core);
            auto runtime_args_0 = get_runtime_args_for_given_ranges(
                physical_core_coords,
                page_stride_vector,
                0,
                input.buffer()->address(),
                0,
                tt::div_up(page_stride_vector.size(), 2));
            auto output_page_offset =
                runtime_args_0[physical_core_coords.size() + 1];  // offset is equivalent to number of pages output in
                                                                  // previous risc core
            rt_config_map_0[core] = runtime_args_0;
            auto runtime_args_1 = get_runtime_args_for_given_ranges(
                physical_core_coords,
                page_stride_vector,
                output_page_offset,
                input.buffer()->address(),
                tt::div_up(page_stride_vector.size(), 2),
                page_stride_vector.size());
            rt_config_map_1[core] = runtime_args_1;
        }
        bool rt_gt_256 = false;
        for (const auto& [core, rt_args] : rt_config_map_0) {
            if (rt_args.size() > 256 || rt_config_map_1[core].size() > 256) {
                rt_gt_256 = true;
                break;
            }
        }
        if (rt_gt_256) {
            auto runtime_args_tensor_0 = construct_per_core_host_tensor(rt_config_map_0);
            auto runtime_args_tensor_1 = construct_per_core_host_tensor(rt_config_map_1);

            auto device_runtime_args_0 =
                move_per_core_config_to_device(runtime_args_tensor_0, output_shard_spec.grid, device);
            auto device_runtime_args_1 =
                move_per_core_config_to_device(runtime_args_tensor_1, output_shard_spec.grid, device);
            inputs.push_back(device_runtime_args_0);
            inputs.push_back(device_runtime_args_1);
        }
    }
    // deallocate the intermediate tensor used to generate rt args
    output_tensor.deallocate();
    return operation::run(
               ReshardDeviceOperation{.output_mem_config = memory_config}, inputs, {}, {optional_output_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
