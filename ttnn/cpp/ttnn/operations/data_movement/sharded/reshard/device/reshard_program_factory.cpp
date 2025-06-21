// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

std::unordered_map<CoreCoord, std::vector<PageStride>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer) {
    const auto& output_buffer_page_mapping = *output_buffer->get_buffer_page_mapping();
    const auto& input_buffer_page_mapping = *input_buffer->get_buffer_page_mapping();

    std::vector<std::pair<CoreCoord, uint32_t>> host_page_to_input_core_mapping(input_buffer->num_pages());
    for (auto mapped_page : input_buffer_page_mapping) {
        auto core = input_buffer_page_mapping.all_cores[mapped_page.core_id];
        host_page_to_input_core_mapping[mapped_page.host_page] = {core, mapped_page.device_page};
    }

    auto output_cores = output_buffer_page_mapping.all_cores;
    // First get output_core to vector< pair<input_core, input_page> (num_pages_in_output)
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

    // now compress to output_core to vector<pair<input_core, input_page_range> (num_page_ranges_in_output)
    std::unordered_map<CoreCoord, std::vector<PageStride>> ret_map;
    ret_map.reserve(output_cores.size());

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

template <bool is_reader>
operation::ProgramWithCallbacks reshard_multi_core_same_width(const Tensor& input, Tensor& output) {
    auto device = input.device();

    tt::tt_metal::Program program{};

    const auto& local_tensor = is_reader ? output : input;
    const auto& remote_tensor = is_reader ? input : output;

    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();
    const auto& all_cores = local_shard_spec.grid;

    auto local_core_type = local_tensor.buffer()->core_type();
    auto remote_core_type = remote_tensor.buffer()->core_type();
    constexpr uint32_t cb_index = tt::CBIndex::c_0;
    constexpr uint32_t cb_scratch_index = tt::CBIndex::c_1;
    auto local_cores = corerange_to_cores(
        local_shard_spec.grid, std::nullopt, local_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto remote_cores = corerange_to_cores(
        remote_shard_spec.grid, std::nullopt, remote_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t unit_size, local_units_per_shard, remote_units_per_shard;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(local_tensor.dtype());

    uint32_t num_units = local_tensor.buffer()->num_pages();
    if (local_tensor.layout() == Layout::TILE) {
        unit_size = tt::tt_metal::detail::TileSize(data_format);
        local_units_per_shard = local_shard_spec.numel() / TILE_HW;
        remote_units_per_shard = remote_shard_spec.numel() / TILE_HW;
    } else {
        unit_size = local_shard_spec.shape[1] * local_tensor.element_size();
        local_units_per_shard = local_shard_spec.shape[0];
        remote_units_per_shard = remote_shard_spec.shape[0];
    }
    uint32_t local_unit_size_padded = tt::align(unit_size, local_tensor.buffer()->alignment());
    uint32_t remote_unit_size_padded = tt::align(unit_size, remote_tensor.buffer()->alignment());
    bool unaligned = false;
    if (remote_unit_size_padded != unit_size || local_unit_size_padded != unit_size) {
        unaligned = true;
    }
    const uint32_t total_size = std::min(local_units_per_shard, remote_units_per_shard) * unit_size;
    const std::string kernel_name =
        is_reader
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_reader.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_writer.cpp";

    bool interface_with_dram = (remote_core_type == CoreType::DRAM);
    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program,
        kernel_name,
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(
            {cb_index,
             interface_with_dram,
             unaligned,
             unit_size,
             local_unit_size_padded,
             remote_unit_size_padded,
             cb_scratch_index}));

    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program,
        kernel_name,
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(
            {cb_index,
             interface_with_dram,
             unaligned,
             unit_size,
             local_unit_size_padded,
             remote_unit_size_padded,
             cb_scratch_index}));

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{cb_index, data_format}})
            .set_page_size(cb_index, unit_size)
            .set_globally_allocated_address(*local_tensor.buffer());
    auto cb_0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    if (unaligned) {
        tt::tt_metal::CircularBufferConfig cb_scratch_config =
            tt::tt_metal::CircularBufferConfig(
                remote_units_per_shard * remote_unit_size_padded, {{cb_scratch_index, data_format}})
                .set_page_size(cb_scratch_index, unit_size);
        auto cb_scratch = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scratch_config);
    }

    uint32_t remote_core_idx = 0;
    uint32_t remote_core_units_rem = remote_units_per_shard;
    uint32_t remote_address = remote_tensor.buffer()->address();
    auto remote_buffer_type = remote_tensor.buffer()->buffer_type();
    auto bank_id =
        device->allocator()->get_bank_ids_from_logical_core(remote_buffer_type, remote_cores[remote_core_idx])[0];

    std::array<tt::tt_metal::KernelHandle, 2> kernels = {kernel_id_0, kernel_id_1};
    uint32_t local_units_left = num_units;
    for (const auto& core : local_cores) {
        uint32_t local_units_per_core = std::min(local_units_left, local_units_per_shard);
        local_units_left -= local_units_per_core;
        uint32_t local_units_per_kernel = tt::div_up(local_units_per_core, kernels.size());
        uint32_t local_start_offset = 0;
        for (const auto& kernel_id : kernels) {
            std::vector<uint32_t> kernel_args = {remote_address, 0, 0};
            uint32_t local_units_to_transfer = std::min(local_units_per_core, local_units_per_kernel);
            if (local_units_to_transfer != 0) {
                uint32_t num_transfers = 0;
                kernel_args[1] = local_start_offset;
                local_start_offset += local_units_to_transfer * unit_size;
                while (local_units_to_transfer > 0) {
                    if (remote_core_units_rem == 0) {
                        remote_core_idx++;
                        remote_core_units_rem = remote_units_per_shard;
                        bank_id = device->allocator()->get_bank_ids_from_logical_core(
                            remote_buffer_type, remote_cores[remote_core_idx])[0];
                    }
                    uint32_t units_to_transfer = std::min(remote_core_units_rem, local_units_to_transfer);
                    bank_id = device->allocator()->get_bank_ids_from_logical_core(
                        remote_buffer_type, remote_cores[remote_core_idx])[0];
                    kernel_args.insert(
                        kernel_args.end(),
                        {bank_id,
                         (remote_units_per_shard - remote_core_units_rem) * remote_unit_size_padded,
                         units_to_transfer});
                    local_units_per_core -= units_to_transfer;
                    local_units_to_transfer -= units_to_transfer;
                    remote_core_units_rem -= units_to_transfer;
                    num_transfers++;
                }
                kernel_args[2] = num_transfers;
            }
            SetRuntimeArgs(program, kernel_id, core, kernel_args);
        }
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_0, local_cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        const auto& local_tensor = is_reader ? output : input;
        const auto& remote_tensor = is_reader ? input : output;
        uint32_t remote_addr = remote_tensor.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : local_cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[0] = remote_addr;
            runtime_args_1[0] = remote_addr;
        }
        UpdateDynamicCircularBufferAddress(program, cb_0, *local_tensor.buffer());
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
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    if (input.layout() == Layout::TILE) {
        page_size = tt::tt_metal::detail::TileSize(data_format);
        unit_size = page_size;
        total_size = output_shard_spec.numel() / TILE_HW * unit_size;
    } else {
        unit_size = output_shard_spec.shape[1] * output.element_size();
        page_size = output.padded_shape()[-1] * output.element_size();
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
        auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(i, 0), input_core_type);
        physical_core_coords.push_back(physical_input_core.x);
    }
    for (uint32_t i = 0; i < grid.y; i++) {
        auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(0, i), input_core_type);
        physical_core_coords.push_back(physical_input_core.y);
    }

    for (const auto& core : cores) {
        auto page_stride_vector = output_core_to_page_range_pair.at(core);
        uint32_t num_ranges = page_stride_vector.size();
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

template <bool is_reader>
operation::ProgramWithCallbacks reshard_multi_core_same_height(const Tensor& input, Tensor& output) {
    auto device = input.device();

    tt::tt_metal::Program program{};

    const auto& local_tensor = is_reader ? output : input;
    const auto& remote_tensor = is_reader ? input : output;

    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();
    const auto& all_cores = local_shard_spec.grid;

    const auto local_core_type = local_tensor.buffer()->core_type();
    const auto remote_core_type = remote_tensor.buffer()->core_type();
    bool interface_with_dram = (remote_core_type == CoreType::DRAM);
    const auto local_cores = corerange_to_cores(
        local_shard_spec.grid, std::nullopt, local_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    const auto remote_cores = corerange_to_cores(
        remote_shard_spec.grid, std::nullopt, remote_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(local_tensor.dtype());
    const uint32_t element_size = tt::datum_size(data_format);

    TT_FATAL(local_tensor.layout() == Layout::ROW_MAJOR, "Expected row major tensor");
    const uint32_t unit_size = local_shard_spec.shape[1] * local_tensor.element_size();  // width * element size
    const uint32_t local_units_per_shard = local_shard_spec.shape[0];                    // height
    const uint32_t remote_units_per_shard = remote_shard_spec.shape[0];                  // height
    const uint32_t total_size = remote_units_per_shard * unit_size;

    constexpr uint32_t cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{cb_index, data_format}})
            .set_page_size(cb_index, unit_size)
            .set_globally_allocated_address(*local_tensor.buffer());
    auto cb_0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    const std::string kernel_name =
        is_reader
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_height_reader.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_height_writer.cpp";

    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program, kernel_name, all_cores, tt::tt_metal::ReaderDataMovementConfig({cb_index, interface_with_dram}));

    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program, kernel_name, all_cores, tt::tt_metal::WriterDataMovementConfig({cb_index, interface_with_dram}));

    uint32_t remote_address = remote_tensor.buffer()->address();
    auto remote_buffer_type = remote_tensor.buffer()->buffer_type();

    // Generate all read/write offsets for each core
    auto [runtime_args_for_each_core, total_num_sticks, local_stride_bytes, remote_stride_bytes] =
        compute_width_sharding_reshard_segments(
            local_shard_spec.shape,
            remote_shard_spec.shape,
            local_cores,
            remote_cores,
            remote_buffer_type,
            remote_core_type,
            device,
            element_size);  // local_core_idx -> runtime args[]

    // Split work across each kernel along tensor height since this is the best way to split work evenly
    const uint32_t total_num_sticks_kernel_0 = total_num_sticks / 2;
    const uint32_t total_num_sticks_kernel_1 = total_num_sticks - total_num_sticks_kernel_0;

    // Here all we do is convert pre-computed offsets into vectors so they can be passed as runtime arguments
    for (uint32_t core_idx = 0; core_idx < local_cores.size(); core_idx++) {
        const auto& args_for_all_segments = runtime_args_for_each_core[core_idx];
        std::vector<uint32_t> runtime_args_0 = {
            total_num_sticks_kernel_0,
            local_stride_bytes,
            remote_stride_bytes,
            remote_address,
            args_for_all_segments.size()};
        std::vector<uint32_t> runtime_args_1 = {
            total_num_sticks_kernel_1,
            local_stride_bytes,
            remote_stride_bytes,
            remote_address,
            args_for_all_segments.size()};
        for (const auto& args : args_for_all_segments) {
            const std::vector<uint32_t> segment_kernel_0 = {
                args.write_size, args.read_offset, args.bank_id, args.write_offset};
            runtime_args_0.insert(runtime_args_0.end(), segment_kernel_0.begin(), segment_kernel_0.end());

            // Adjust read and write offsets to the correct stick address because we are splitting work across 2 kernels
            const uint32_t adjusted_read_offset = args.read_offset + total_num_sticks_kernel_0 * local_stride_bytes;
            const uint32_t adjusted_write_offset = args.write_offset + total_num_sticks_kernel_0 * remote_stride_bytes;

            const std::vector<uint32_t> segment_kernel_1 = {
                args.write_size, adjusted_read_offset, args.bank_id, adjusted_write_offset};
            runtime_args_1.insert(runtime_args_1.end(), segment_kernel_1.begin(), segment_kernel_1.end());
        }
        SetRuntimeArgs(program, kernel_id_0, local_cores[core_idx], runtime_args_0);
        SetRuntimeArgs(program, kernel_id_1, local_cores[core_idx], runtime_args_1);
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_0, local_cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        const auto& local_tensor = is_reader ? output : input;
        const auto& remote_tensor = is_reader ? input : output;
        uint32_t remote_address = remote_tensor.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : local_cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[3] = remote_address;
            runtime_args_1[3] = remote_address;
        }
        UpdateDynamicCircularBufferAddress(program, cb_0, *local_tensor.buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks reshard_multi_core(const Tensor& input, Tensor& output) {
    if (input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
        output.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        if (output.memory_config().buffer_type() == BufferType::L1) {
            return reshard_multi_core_same_width<true>(input, output);
        } else {
            return reshard_multi_core_same_width<false>(input, output);
        }
    } else if (
        input.layout() == Layout::ROW_MAJOR &&
        input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
        output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        if (output.memory_config().buffer_type() == BufferType::L1) {
            return reshard_multi_core_same_height<true>(input, output);
        } else {
            return reshard_multi_core_same_height<false>(input, output);
        }
    } else {
        return reshard_multi_core_generic(input, output);
    }
}

}  // namespace ttnn::operations::data_movement::detail
