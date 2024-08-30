// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "reshard_program_factory.hpp"
using namespace tt::constants;


namespace ttnn::operations::data_movement::detail {

std::unordered_map<CoreCoord, std::vector<PageStride>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer) {
    auto output_buffer_page_mapping = generate_buffer_page_mapping(*output_buffer);
    auto input_buffer_page_mapping = generate_buffer_page_mapping(*input_buffer);

    const auto& output_shard_to_host_mapping = output_buffer_page_mapping.dev_page_to_host_page_mapping_;
    const auto& input_page_to_local_page_mapping = input_buffer_page_mapping.host_page_to_local_shard_page_mapping_;
    const auto& host_page_to_input_page_mapping = input_buffer_page_mapping.host_page_to_dev_page_mapping_;

    auto output_cores = output_buffer_page_mapping.all_cores_;
    // First get output_core to vector< pair<input_core, input_page> (num_pages_in_output)
    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page(
        output_cores.size());

    for (uint32_t output_page_id = 0; output_page_id < output_buffer->num_dev_pages(); output_page_id++) {
        auto output_core_id = output_buffer_page_mapping.dev_page_to_core_mapping_[output_page_id];
        TT_ASSERT(output_core_id < output_cores.size());
        auto host_page = output_shard_to_host_mapping[output_page_id];
        std::optional<std::pair<CoreCoord, uint32_t>> mapped_page = std::nullopt;
        if (host_page.has_value()) {
            auto input_page = host_page_to_input_page_mapping[host_page.value()];
            auto local_input_page = input_page_to_local_page_mapping[host_page.value()];
            auto input_core =
                input_buffer_page_mapping.all_cores_[input_buffer_page_mapping.dev_page_to_core_mapping_[input_page]];
            mapped_page = std::make_optional<std::pair<CoreCoord, uint32_t>>({input_core, local_input_page});
        }
        output_core_to_vector_input_core_page[output_core_id].push_back(mapped_page);
    }

    // now compress to output_core to vector<pair<input_core, input_page_range> (num_page_ranges_in_output)
    std::unordered_map<CoreCoord, std::vector<PageStride>> ret_map;
    ret_map.reserve(output_cores.size());

    auto output_core_host_page_indices = output_buffer_page_mapping.core_host_page_indices_;
    auto device = input_buffer->device();
    auto full_grid = device->compute_with_storage_grid_size();
    CoreCoord end_core = (*output_buffer->shard_spec().grid().ranges().rbegin()).end_coord;
    uint32_t output_core_id = 0;
    for (auto output_core : output_cores) {
        ret_map.try_emplace(output_core, std::vector<PageStride>{});

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
                ret_map[output_core].push_back(PageStride{
                    .start_core = output_core,
                    .start_data = 0,
                    .stride_size = stride_size,
                    .stride = Stride{.core = {0, 0}, .data = 0},
                    .num_strides = 1,
                    .skip = true});
                it += stride_size;
            } else {
                const auto start_core = it->value().first;
                const auto start_page = it->value().second;
                auto expected_next_page = start_page + 1;
                Stride stride = Stride{.core = {0, 0}, .data = 0};
                if ((it + 1) == end) {
                    ret_map[output_core].push_back(PageStride{
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
                        consecutive_it = consecutive_it + 1;
                        last_it_consec = consecutive_it;
                    }
                    uint32_t stride_size = std::distance(it, last_it_consec);
                    if (last_it_consec == it) {
                        stride_size = 1;
                    }
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
                            stride = Stride{.core = {0, 0}, .data = data_stride};
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
                            stride = Stride{
                                .core =
                                    {next_input_page.value().first.x - prev_input_page.value().first.x,
                                     next_input_page.value().first.y - prev_input_page.value().first.y},
                                .data = data_stride};
                        }
                        // diff data and diff core, not handled yet
                        else {
                            TT_ASSERT(it->has_value());
                            ret_map[output_core].push_back(PageStride{
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
                    auto stride_start = stride_it;
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
                        bool core_stride = ((stride.core.x != 0) or (stride.core.y != 0));
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
                    ret_map[output_core].push_back(PageStride{
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

enum class ReshardStridesInRange { ALL_STRIDES, FIRST_HALF, SECOND_HALF };

std::vector<uint32_t> get_runtime_args_for_given_ranges(
    const std::vector<uint32_t>& physical_core_coords,
    const std::vector<PageStride>& page_stride_vector,
    const uint32_t output_page_offset,
    const uint32_t& input_addr,
    const uint32_t starting_range,
    const uint32_t ending_range,
    const ReshardStridesInRange reshard_strides_in_range = ReshardStridesInRange::ALL_STRIDES) {
    std::vector<uint32_t> runtime_args = physical_core_coords;
    runtime_args.push_back(input_addr);
    runtime_args.push_back(0);
    runtime_args.push_back(ending_range - starting_range);
    runtime_args.push_back(output_page_offset);
    uint32_t num_output_pages = 0;

    for (uint32_t range_id = starting_range; range_id < ending_range; range_id++) {
        PageStride ps = page_stride_vector[range_id];
        uint32_t num_strides;
        uint32_t start_core_x;
        uint32_t start_core_y;
        uint32_t start_data;
        if (reshard_strides_in_range == ReshardStridesInRange::ALL_STRIDES) {
            num_strides = ps.num_strides;
            start_core_x = ps.start_core.x;
            start_core_y = ps.start_core.y;
            start_data = ps.start_data;
        } else {
            if (reshard_strides_in_range == ReshardStridesInRange::FIRST_HALF) {
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

operation::ProgramWithCallbacks reshard_multi_core_same_width(const Tensor& input, Tensor& output) {
    auto device = input.device();

    tt::tt_metal::Program program{};

    const auto input_shard_spec = input.shard_spec().value();
    const auto output_shard_spec = output.shard_spec().value();
    const auto& all_cores = output_shard_spec.grid;
    auto grid = input.buffer()->buffer_type() == BufferType::DRAM ? device->dram_grid_size()
                                                                  : device->compute_with_storage_grid_size();
    auto input_core_type = input.buffer()->core_type();
    constexpr uint32_t dst_cb_index = tt::CB::c_in0;
    auto input_cores = corerange_to_cores(
        input_shard_spec.grid, std::nullopt, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto output_cores =
        corerange_to_cores(all_cores, std::nullopt, output_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t total_size, unit_size, input_units_per_shard, output_units_per_shard;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    uint32_t num_output_units = input.buffer()->num_pages();
    if (input.get_layout() == Layout::TILE) {
        unit_size = tt::tt_metal::detail::TileSize(data_format);
        input_units_per_shard = input_shard_spec.numel() / TILE_HW;
        output_units_per_shard = output_shard_spec.numel() / TILE_HW;
        total_size = output_units_per_shard * unit_size;
    } else {
        unit_size = output_shard_spec.shape[1] * output.element_size();
        input_units_per_shard = input_shard_spec.shape[0];
        output_units_per_shard = output_shard_spec.shape[0];
        total_size = output_units_per_shard * unit_size;
    }

    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig({dst_cb_index}));

    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_reader.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig({dst_cb_index}));

    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{dst_cb_index, data_format}})
            .set_page_size(dst_cb_index, unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    uint32_t input_core_idx = 0;
    uint32_t input_core_units_rem = input_units_per_shard;
    uint32_t input_address = input.buffer()->address();
    auto input_buffer_type = input.buffer()->buffer_type();
    auto bank_id = device->bank_ids_from_logical_core(input_buffer_type, input_cores[input_core_idx])[0];
    uint32_t bank_offset = device->bank_offset(input_buffer_type, bank_id);
    auto input_core = device->physical_core_from_logical_core(input_cores[input_core_idx], input_core_type);

    std::array<tt::tt_metal::KernelHandle, 2> kernels = {kernel_id_0, kernel_id_1};
    uint32_t output_units_left = num_output_units;
    for (const auto& core : output_cores) {
        uint32_t output_units_per_core = std::min(output_units_left, output_units_per_shard);
        output_units_left -= output_units_per_core;
        uint32_t output_units_per_kernel = tt::div_up(output_units_per_core, kernels.size());
        uint32_t output_start_offset = 0;
        for (const auto& kernel_id : kernels) {
            std::vector<uint32_t> kernel_args = {input_address, 0, 0};
            uint32_t output_units_to_get = std::min(output_units_per_core, output_units_per_kernel);
            if (output_units_to_get != 0) {
                uint32_t num_reads = 0;
                kernel_args[1] = output_start_offset;
                output_start_offset += output_units_to_get * unit_size;
                while (output_units_to_get > 0) {
                    if (input_core_units_rem == 0) {
                        input_core_idx++;
                        input_core_units_rem = input_units_per_shard;
                        bank_id = device->bank_ids_from_logical_core(input_buffer_type, input_cores[input_core_idx])[0];
                        bank_offset = device->bank_offset(input_buffer_type, bank_id);
                        input_core = device->physical_core_from_logical_core(input_cores[input_core_idx], input_core_type);
                    }
                    uint32_t units_to_read = std::min(input_core_units_rem, output_units_to_get);
                    auto input_core =
                        device->physical_core_from_logical_core(input_cores[input_core_idx], input_core_type);
                    kernel_args.insert(
                        kernel_args.end(),
                        {static_cast<uint32_t>(input_core.x),
                         static_cast<uint32_t>(input_core.y),
                         (input_units_per_shard - input_core_units_rem) * unit_size + bank_offset,
                         units_to_read * unit_size});
                    output_units_per_core -= units_to_read;
                    output_units_to_get -= units_to_read;
                    input_core_units_rem -= units_to_read;
                    num_reads++;
                }
                kernel_args[2] = num_reads;
            }
            SetRuntimeArgs(program, kernel_id, core, kernel_args);
        }
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_dst0, grid, output_cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        uint32_t input_addr = input.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : output_cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[0] = input_addr;
            runtime_args_1[0] = input_addr;
        }
        UpdateDynamicCircularBufferAddress(program, cb_dst0, *output.buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks reshard_multi_core_generic(const Tensor& input, Tensor& output) {
    auto device = input.device();
    auto output_core_to_page_range_pair = get_core_page_ranges(input.buffer(), output.buffer());

    tt::tt_metal::Program program{};

    auto input_shard_spec = input.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();
    auto all_cores = output_shard_spec.grid;
    auto grid = input.buffer()->buffer_type() == BufferType::DRAM ? device->dram_grid_size()
                                                                  : device->compute_with_storage_grid_size();
    auto input_core_type = input.buffer()->core_type();
    uint32_t dst_cb_index = 16;
    auto cores =
        corerange_to_cores(all_cores, std::nullopt, output_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t total_size, page_size, unit_size;
    auto output_shard_shape = output_shard_spec.shape;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    if (input.get_layout() == Layout::TILE) {
        page_size = tt::tt_metal::detail::TileSize(data_format);
        unit_size = page_size;
        total_size = output_shard_spec.numel() / TILE_HW * unit_size;
    } else {
        unit_size = output_shard_spec.shape[1] * output.element_size();
        page_size = output.get_legacy_shape()[-1] * output.element_size();
        total_size = output_shard_shape[0] * unit_size;
    }

    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig({dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size}));

    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig({dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size}));

    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{dst_cb_index, data_format}})
            .set_page_size(dst_cb_index, unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    std::vector<uint32_t> physical_core_coords;
    physical_core_coords.reserve(grid.x * grid.y);
    for (uint32_t i = 0; i < grid.x; i++) {
        auto physical_input_core = device->physical_core_from_logical_core(CoreCoord(i, 0), input_core_type);
        physical_core_coords.push_back(physical_input_core.x);
    }
    for (uint32_t i = 0; i < grid.y; i++) {
        auto physical_input_core = device->physical_core_from_logical_core(CoreCoord(0, i), input_core_type);
        physical_core_coords.push_back(physical_input_core.y);
    }

    for (const auto& core : cores) {
        auto page_stride_vector = output_core_to_page_range_pair.at(core);
        uint32_t num_ranges = page_stride_vector.size();
        std::vector<uint32_t> runtime_args = physical_core_coords;
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
        tt::tt_metal::SetRuntimeArgs(program, kernel_id_0, core, runtime_args_0);
        auto runtime_args_1 = get_runtime_args_for_given_ranges(
            physical_core_coords,
            page_stride_vector,
            output_page_offset,
            input.buffer()->address(),
            tt::div_up(page_stride_vector.size(), 2),
            page_stride_vector.size());
        tt::tt_metal::SetRuntimeArgs(program, kernel_id_1, core, runtime_args_1);
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_dst0, grid, cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        uint32_t input_addr = input.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[grid.x + grid.y] = input_addr;
            runtime_args_1[grid.x + grid.y] = input_addr;
        }
        UpdateDynamicCircularBufferAddress(program, cb_dst0, *output.buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks reshard_multi_core(const Tensor& input, Tensor& output) {
    if (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
        output.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        return reshard_multi_core_same_width(input, output);
    } else {
        return reshard_multi_core_generic(input, output);
    }
}

}  // namespace ttnn::operations::data_movement::detail
