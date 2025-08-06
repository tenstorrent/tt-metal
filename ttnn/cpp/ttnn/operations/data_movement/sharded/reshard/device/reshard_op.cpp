// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_op.hpp"

#include <enchantum/enchantum.hpp>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

#include "reshard_program_factory.hpp"
#include "nd_reshard_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

namespace CMAKE_UNIQUE_NAMESPACE {
bool is_valid_for_legacy_reshard(const Tensor& input_tensor, const MemoryConfig& out_mem_config) {
    auto inp_mem_layout = input_tensor.memory_config().memory_layout();
    auto out_mem_layout = out_mem_config.memory_layout();

    auto inp_buffer_type = input_tensor.memory_config().buffer_type();
    auto out_buffer_type = out_mem_config.buffer_type();

    if (!input_tensor.memory_config().shard_spec().has_value() || !out_mem_config.shard_spec().has_value()) {
        // If shard_spec has no value, then we can only use nd resharding
        return false;
    }

    if (inp_mem_layout == out_mem_layout && inp_mem_layout != TensorMemoryLayout::BLOCK_SHARDED) {
        // Resharding must have at least one buffer in L1
        return inp_buffer_type == BufferType::L1 || out_buffer_type == BufferType::L1;
    } else {
        // Resharding requires output buffer to be in L1
        return out_mem_config.buffer_type() == BufferType::L1;
    }

    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        if (inp_mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            // row major must have shard_spec[0] be the same on both input and output
            return input_tensor.memory_config().shard_spec().value().shape[0] ==
                   out_mem_config.shard_spec().value().shape[0];
        } else {
            // row major must have shard_spec[1] be the same on both input and output
            return input_tensor.memory_config().shard_spec().value().shape[1] ==
                   out_mem_config.shard_spec().value().shape[1];
        }
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE

void ReshardDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.is_sharded(), "input must be sharded");

    bool has_output_tensor = output_tensors.size() == 1 && output_tensors[0].has_value();
    if (has_output_tensor) {
        const auto& output_tensor = output_tensors[0].value();
        TT_FATAL(input_tensor.logical_shape() == output_tensor.logical_shape(), "Error");
        TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Error");
        TT_FATAL(input_tensor.layout() == output_tensor.layout(), "Error");
    }
    const auto& out_mem_config =
        has_output_tensor ? output_tensors[0].value().memory_config() : this->output_mem_config;
    TT_FATAL(out_mem_config.is_sharded(), "output must be sharded");

    if (!CMAKE_UNIQUE_NAMESPACE::is_valid_for_legacy_reshard(input_tensor, out_mem_config)) {
        auto output_tensor_spec = compute_output_specs(input_tensors, output_tensors).front();
        auto out_distribution_spec = output_tensor_spec.compute_buffer_sharding_args().buffer_distribution_spec();
        auto input_distribution_spec = input_tensor.buffer()->buffer_distribution_spec();

        auto n_logical_input_pages = input_distribution_spec->tensor_shape_in_pages().volume();
        auto n_logical_output_pages = out_distribution_spec->tensor_shape_in_pages().volume();

        auto input_page_size = input_tensor.tensor_spec().compute_page_size_bytes();
        auto output_page_size = output_tensor_spec.compute_page_size_bytes();

        TT_FATAL(
            n_logical_input_pages == n_logical_output_pages,
            "Number of input ({}) and output ({}) pages must match",
            n_logical_input_pages,
            n_logical_output_pages);
        TT_FATAL(
            input_page_size == output_page_size,
            "Input and output tensors must have the same page size. Input page size: {}, Output page size: {}",
            input_page_size,
            output_page_size);
        TT_FATAL(
            input_tensor.layout() == output_tensor_spec.tensor_layout().get_layout(),
            "Input and output tensors must have the same layout. Input layout: {}, Output layout: {}",
            enchantum::to_string(input_tensor.layout()),
            enchantum::to_string(output_tensor_spec.tensor_layout().get_layout()));
    }
}

std::vector<ttnn::TensorSpec> ReshardDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0]->tensor_spec()};
    }

    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.layout(),
            output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
ReshardDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
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

Tensor construct_empty_rt_tensor(std::vector<CoreCoord> output_cores, uint32_t MAX_RT_ARGS_WIDTH) {
    // Sort cores to ensure consistent ordering
    std::sort(output_cores.begin(), output_cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return (a.y < b.y) || (a.y == b.y && a.x < b.x);
    });

    // Create shape based on number of cores and width
    ttnn::Shape tensor_shape({static_cast<uint32_t>(output_cores.size()), MAX_RT_ARGS_WIDTH});

    // Create zeroed data vector
    std::vector<uint32_t> flattened_data(output_cores.size() * MAX_RT_ARGS_WIDTH, 0);

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

operation::ProgramWithCallbacks ReshardDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    Tensor local_rt_args_config_0, local_rt_args_config_1;
    std::unordered_map<CoreCoord, std::vector<detail::PageStride>> output_core_to_page_range_pair;
    uint32_t MAX_RT_ARGS_WIDTH = 0;
    bool not_generic =
        (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
         output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED);
    not_generic = not_generic || (input_tensor.layout() == Layout::ROW_MAJOR &&
                                  input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                                  output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);

    if (!not_generic) {
        if (input_tensor.buffer()->page_size() != output_tensor.buffer()->page_size()) {
            output_core_to_page_range_pair =
                get_core_page_ranges_diff_width(input_tensor.buffer(), output_tensor.buffer(), input_tensor);
        } else {
            output_core_to_page_range_pair = get_core_page_ranges(input_tensor.buffer(), output_tensor.buffer());
        }
    }
    if (output_tensor.shard_spec().has_value()) {
        auto output_grid = output_tensor.shard_spec().value().grid;
        auto output_cores = corerange_to_cores(output_grid);

        // Estimate RT args needed
        uint32_t max_ranges = 0;
        for (const auto& [core, page_strides] : output_core_to_page_range_pair) {
            if (page_strides.size() > max_ranges) {
                max_ranges = page_strides.size();
            }
        }
        // Estimate rt args needed
        uint32_t max_rt_args = 10000;  // before L1 buffers start to clash
        // for each range, we have around 3 rt args, and 4 rt args for header
        MAX_RT_ARGS_WIDTH = std::min(max_ranges * 3 + 4, max_rt_args);
        auto rt_args_host_0 = construct_empty_rt_tensor(output_cores, MAX_RT_ARGS_WIDTH);
        auto rt_args_host_1 = construct_empty_rt_tensor(output_cores, MAX_RT_ARGS_WIDTH);

        local_rt_args_config_0 = move_per_core_config_to_device(rt_args_host_0, output_grid, output_tensor.device());
        local_rt_args_config_1 = move_per_core_config_to_device(rt_args_host_1, output_grid, output_tensor.device());
        // Using default RT args approach in this case
        if (MAX_RT_ARGS_WIDTH < 256) {
            local_rt_args_config_0 = Tensor();
            local_rt_args_config_1 = Tensor();
        }
    }
    if (CMAKE_UNIQUE_NAMESPACE::is_valid_for_legacy_reshard(input_tensor, output_tensor.memory_config())) {
        return detail::reshard_multi_core(
            input_tensor,
            output_tensor,
            local_rt_args_config_0,
            local_rt_args_config_1,
            output_core_to_page_range_pair,
            MAX_RT_ARGS_WIDTH);
    } else {
        return detail::nd_reshard_multi_core(input_tensor, output_tensor);
    }
}

std::vector<Tensor> ReshardDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors.at(0).device())};
}

}  // namespace ttnn::operations::data_movement
