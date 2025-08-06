// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_program_factory.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/reshard.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

template <bool is_reader>
operation::ProgramWithCallbacks reshard_multi_core_same_width(const Tensor& input, Tensor& output) {
    auto device = input.device();

    tt::tt_metal::Program program{};

    const auto& local_tensor = is_reader ? output : input;
    const auto& remote_tensor = is_reader ? input : output;

    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();
    const auto& all_cores = local_shard_spec.grid;

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
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scratch_config);
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

operation::ProgramWithCallbacks reshard_multi_core_generic(const std::vector<Tensor>& inputs, Tensor& output) {
    auto input = inputs.at(0);
    auto device = input.device();

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
        // For ROW_MAJOR, use base page size from GCD calculation
        uint32_t input_page_size = input.buffer()->page_size();
        uint32_t output_page_size = output.buffer()->page_size();
        uint32_t base_page_size = std::gcd(input_page_size, output_page_size);

        unit_size = base_page_size;
        page_size = base_page_size;
        total_size = output_shard_shape[0] * output_shard_shape[1] * output.element_size();
    }

    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{dst_cb_index, data_format}})
            .set_page_size(dst_cb_index, output.buffer()->page_size())
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    tt::tt_metal::KernelHandle kernel_id_0;
    tt::tt_metal::KernelHandle kernel_id_1;
    bool rt_gt_256 = false;

    std::unordered_map<CoreCoord, std::vector<detail::PageStride>> output_core_to_page_range_pair;
    if (input.buffer()->page_size() != output.buffer()->page_size()) {
        output_core_to_page_range_pair = get_core_page_ranges_diff_width(input.buffer(), output.buffer(), input);
    } else {
        output_core_to_page_range_pair = get_core_page_ranges(input.buffer(), output.buffer());
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

    std::unordered_map<CoreCoord, std::vector<uint32_t>> rt_config_map_0;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> rt_config_map_1;
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
    for (const auto& [core, rt_args] : rt_config_map_0) {
        if (rt_args.size() > 256 || rt_config_map_1[core].size() > 256) {
            rt_gt_256 = true;
            break;
        }
    }
    CBHandle cb_rt_args_0 = 0;
    CBHandle cb_rt_args_1 = 0;
    if (rt_gt_256) {
        auto device_runtime_args_0 = inputs.at(1);
        auto device_runtime_args_1 = inputs.at(2);
        constexpr uint32_t rt_args_cb_index_0 = 17;
        constexpr uint32_t rt_args_cb_index_1 = 18;

        // CB config for first runtime args tensor
        auto cb_rt_args_config_0 =
            tt::tt_metal::CircularBufferConfig(
                device_runtime_args_0.logical_shape()[1] * sizeof(uint32_t),
                {{rt_args_cb_index_0, tt::DataFormat::Int32}})
                .set_page_size(rt_args_cb_index_0, device_runtime_args_0.logical_shape()[1] * sizeof(uint32_t))
                .set_globally_allocated_address(*device_runtime_args_0.buffer());

        // CB config for second runtime args tensor
        auto cb_rt_args_config_1 =
            tt::tt_metal::CircularBufferConfig(
                device_runtime_args_1.logical_shape()[1] * sizeof(uint32_t),
                {{rt_args_cb_index_1, tt::DataFormat::Int32}})
                .set_page_size(rt_args_cb_index_1, device_runtime_args_1.logical_shape()[1] * sizeof(uint32_t))
                .set_globally_allocated_address(*device_runtime_args_1.buffer());

        // Create the circular buffers
        cb_rt_args_0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_rt_args_config_0);
        cb_rt_args_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_rt_args_config_1);

        kernel_id_0 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/reshard_reader_tensor_rt.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(
                {dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size, unit_size, rt_args_cb_index_0}));

        kernel_id_1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/reshard_reader_tensor_rt.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(
                {dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size, unit_size, rt_args_cb_index_1}));
        for (const auto& core : cores) {
            // Set runtime args for each core
            tt::tt_metal::SetRuntimeArgs(program, kernel_id_0, core, {input.buffer()->address()});
            tt::tt_metal::SetRuntimeArgs(program, kernel_id_1, core, {input.buffer()->address()});
        }

    } else {
        kernel_id_0 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(
                {dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size, unit_size}));

        kernel_id_1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(
                {dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size, unit_size}));

        for (const auto& core : cores) {
            // Set runtime args for each core
            auto rt_args_0 = rt_config_map_0[core];
            auto rt_args_1 = rt_config_map_1[core];
            tt::tt_metal::SetRuntimeArgs(program, kernel_id_0, core, rt_args_0);
            tt::tt_metal::SetRuntimeArgs(program, kernel_id_1, core, rt_args_1);
        }
    }
    auto override_runtime_arguments_callback =
        [kernel_id_0, kernel_id_1, rt_gt_256, cb_dst0, cb_rt_args_0, cb_rt_args_1, grid, cores](
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
            if (rt_gt_256 == false) {
                for (auto core : cores) {
                    auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
                    auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
                    runtime_args_0[grid.x + grid.y] = input_addr;
                    runtime_args_1[grid.x + grid.y] = input_addr;
                }
            } else {
                const auto& tensor_rt_args0 = input_tensors.at(1);
                const auto& tensor_rt_args1 = input_tensors.at(2);
                UpdateDynamicCircularBufferAddress(program, cb_rt_args_0, *tensor_rt_args0.buffer());
                UpdateDynamicCircularBufferAddress(program, cb_rt_args_1, *tensor_rt_args1.buffer());
                for (auto core : cores) {
                    auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
                    auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
                    runtime_args_0[0] = input_addr;
                    runtime_args_1[0] = input_addr;
                }
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

operation::ProgramWithCallbacks reshard_multi_core(const std::vector<Tensor>& inputs, Tensor& output) {
    auto input = inputs.at(0);
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
        return reshard_multi_core_generic(inputs, output);
    }
}

}  // namespace ttnn::operations::data_movement::detail
