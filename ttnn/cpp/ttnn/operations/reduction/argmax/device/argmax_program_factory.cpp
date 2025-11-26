// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_program_factory.hpp"

#include <string>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/math.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::argmax::program {

using namespace tt::constants;

namespace {

// Distributes work across cores for argmax reduction operations
static inline std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> distribute_work_to_cores(
    const IDevice* device,
    uint32_t red_dim_units,
    uint32_t min_red_dim_units_per_core,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    CoreRangeSet all_cores, cores0, cores1;
    uint32_t red_dim_units0, red_dim_units1;

    if (sub_core_grids.has_value()) {
        all_cores = sub_core_grids.value();
        const uint32_t total_blocks = tt::div_up(red_dim_units, min_red_dim_units_per_core);

        if (all_cores.size() == 2) {
            cores0 = CoreRangeSet(all_cores.ranges().at(0));
            cores1 = CoreRangeSet(all_cores.ranges().at(1));

            const uint32_t total_cores = cores0.num_cores() + cores1.num_cores();
            const uint32_t blocks_per_core = tt::div_up(total_blocks, total_cores);

            red_dim_units0 = blocks_per_core * min_red_dim_units_per_core;
            red_dim_units1 = blocks_per_core * min_red_dim_units_per_core;
        } else {
            cores0 = all_cores;
            cores1 = CoreRangeSet();

            const auto total_cores = cores0.num_cores();
            const uint32_t blocks_per_core = tt::div_up(total_blocks, total_cores);

            red_dim_units0 = blocks_per_core * min_red_dim_units_per_core;
            red_dim_units1 = 0;
        }
    } else {
        const auto core_grid = device->compute_with_storage_grid_size();
        uint32_t num_total_cores;
        std::tie(num_total_cores, all_cores, cores0, cores1, red_dim_units0, red_dim_units1) =
            split_work_to_cores(core_grid, tt::div_up(red_dim_units, min_red_dim_units_per_core));
        red_dim_units0 *= min_red_dim_units_per_core;
        red_dim_units1 *= min_red_dim_units_per_core;
    }

    return {all_cores, cores0, cores1, red_dim_units0, red_dim_units1};
}

std::tuple<uint32_t, uint32_t> get_page_sizes_single_core(
    const Tensor& input, const Tensor& output, bool keepdim, bool reduce_all) {
    uint32_t src_page_size = 0;
    uint32_t dst_page_size = 0;

    const auto& input_shape = input.padded_shape();
    const uint32_t rank = input_shape.size();

    if (input.layout() == Layout::ROW_MAJOR) {
        const uint32_t red_dim_units = input_shape[rank - 1];
        const uint32_t input_unit_size = input.element_size();
        const uint32_t output_unit_size = output.element_size();
        const auto output_last_dim = reduce_all || keepdim || (rank < 2) ? 1 : input_shape[rank - 2];
        src_page_size = red_dim_units * input_unit_size;
        dst_page_size = output_last_dim * output_unit_size;
    } else {
        src_page_size = input.tensor_spec().compute_page_size_bytes();
        dst_page_size = output.tensor_spec().compute_page_size_bytes();
    }

    return {src_page_size, dst_page_size};
}

void create_circular_buffers_single_core(
    Program& program,
    const CoreRangeSet& all_cores,
    uint32_t src_cb_index,
    uint32_t dst_cb_index,
    uint32_t src_page_size,
    uint32_t dst_page_size,
    tt::DataFormat input_format,
    tt::DataFormat output_format) {
    CircularBufferConfig src_cb_config =
        CircularBufferConfig(src_page_size, {{src_cb_index, input_format}}).set_page_size(src_cb_index, src_page_size);
    CreateCircularBuffer(program, all_cores, src_cb_config);

    const CircularBufferConfig dst_cb_config =
        CircularBufferConfig(dst_page_size, {{dst_cb_index, output_format}}).set_page_size(dst_cb_index, dst_page_size);
    CreateCircularBuffer(program, all_cores, dst_cb_config);
}

std::vector<uint32_t> get_ctime_args_row_major(
    const Tensor& input,
    const Tensor& output,
    uint32_t src_page_size,
    uint32_t dst_page_size,
    uint32_t src_cb_index,
    uint32_t dst_cb_index,
    bool keepdim,
    bool reduce_all) {
    std::vector<uint32_t> ctime_args;

    const auto& input_shape = input.padded_shape();
    const uint32_t rank = input_shape.size();

    const uint32_t red_dim_units = input_shape[rank - 1];
    const auto output_last_dim = reduce_all || keepdim || (rank < 2) ? 1 : input_shape[rank - 2];

    const auto inner_dim_units = output_last_dim;
    const auto outer_dim_units = input.logical_volume() / inner_dim_units / red_dim_units;

    ctime_args.insert(
        ctime_args.end(),
        {src_cb_index,
         dst_cb_index,
         src_page_size,
         dst_page_size,
         outer_dim_units,
         inner_dim_units,
         red_dim_units,
         static_cast<uint32_t>(reduce_all)});

    return ctime_args;
}

std::vector<uint32_t> get_ctime_args_tile(
    const Tensor& input,
    const Tensor& output,
    uint32_t src_page_size,
    uint32_t dst_page_size,
    uint32_t src_cb_index,
    uint32_t dst_cb_index,
    bool keepdim,
    bool reduce_all) {
    std::vector<uint32_t> ctime_args;

    const auto& input_shape = input.padded_shape();
    const uint32_t rank = input_shape.size();

    const uint32_t logical_rank = input.logical_shape().size();
    const uint32_t w_tiles = input_shape[rank - 1] / TILE_WIDTH;
    const uint32_t h_tiles = input_shape[rank - 2] / TILE_HEIGHT;

    const uint32_t w_logical = input.logical_shape()[logical_rank - 1];
    const uint32_t h_logical = logical_rank > 1 ? input.logical_shape()[logical_rank - 2] : 1;

    const uint32_t outer_dim_units = input.logical_volume() / (h_logical * w_logical);

    ctime_args.insert(
        ctime_args.end(),
        {src_cb_index,
         dst_cb_index,
         src_page_size,
         dst_page_size,
         TILE_HEIGHT,
         TILE_WIDTH,
         h_tiles,
         w_tiles,
         h_logical,
         w_logical,
         outer_dim_units,
         static_cast<uint32_t>(reduce_all),
         static_cast<uint32_t>(keepdim)});

    return ctime_args;
}

}  // namespace

// =============================================================================
// ArgMaxSingleCoreRowMajorFactory
// =============================================================================

ArgMaxSingleCoreRowMajorFactory::cached_program_t ArgMaxSingleCoreRowMajorFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    Program program{};

    const auto& input = tensor_args.input;
    const IDevice* device = output.device();
    const bool reduce_all = !args.dim.has_value();

    const auto src_cb_index = tt::CBIndex::c_0;
    const auto dst_cb_index = tt::CBIndex::c_1;

    const auto grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_units = 1;
    auto [num_cores, all_cores, unused_1, unused_2, unused_3, unused_4] = split_work_to_cores(grid_size, num_units);

    const tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    const auto [src_page_size, dst_page_size] = get_page_sizes_single_core(input, output, args.keepdim, reduce_all);

    create_circular_buffers_single_core(
        program,
        all_cores,
        src_cb_index,
        dst_cb_index,
        src_page_size,
        dst_page_size,
        input_data_format,
        output_data_format);

    std::vector<uint32_t> ctime_args = get_ctime_args_row_major(
        input, output, src_page_size, dst_page_size, src_cb_index, dst_cb_index, args.keepdim, reduce_all);

    const auto src_buffer = input.buffer();
    const auto dst_buffer = output.buffer();
    TensorAccessorArgs(src_buffer).append_to(ctime_args);
    TensorAccessorArgs(dst_buffer).append_to(ctime_args);

    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved.cpp";
    const std::map<std::string, std::string> kernel_defines;
    const KernelHandle reader_kernel_id =
        CreateKernel(program, kernel_path, all_cores, ReaderDataMovementConfig(ctime_args, kernel_defines));

    const auto cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, false);
    const CoreCoord core = cores.at(0);
    SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), dst_buffer->address()});

    return {std::move(program), {.reader_kernel_id = reader_kernel_id, .core = core}};
}

void ArgMaxSingleCoreRowMajorFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id, shared_vars.core);
    runtime_args[0] = src_buffer->address();
    runtime_args[1] = dst_buffer->address();
}

// =============================================================================
// ArgMaxSingleCoreTileFactory
// =============================================================================

ArgMaxSingleCoreTileFactory::cached_program_t ArgMaxSingleCoreTileFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    Program program{};

    const auto& input = tensor_args.input;
    const IDevice* device = output.device();
    const bool reduce_all = !args.dim.has_value();

    const auto src_cb_index = tt::CBIndex::c_0;
    const auto dst_cb_index = tt::CBIndex::c_1;

    const auto grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_units = 1;
    auto [num_cores, all_cores, unused_1, unused_2, unused_3, unused_4] = split_work_to_cores(grid_size, num_units);

    const tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    const auto [src_page_size, dst_page_size] = get_page_sizes_single_core(input, output, args.keepdim, reduce_all);

    create_circular_buffers_single_core(
        program,
        all_cores,
        src_cb_index,
        dst_cb_index,
        src_page_size,
        dst_page_size,
        input_data_format,
        output_data_format);

    std::vector<uint32_t> ctime_args = get_ctime_args_tile(
        input, output, src_page_size, dst_page_size, src_cb_index, dst_cb_index, args.keepdim, reduce_all);

    const auto src_buffer = input.buffer();
    const auto dst_buffer = output.buffer();
    TensorAccessorArgs(src_buffer).append_to(ctime_args);
    TensorAccessorArgs(dst_buffer).append_to(ctime_args);

    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_tile_layout.cpp";
    const std::map<std::string, std::string> kernel_defines;
    const KernelHandle reader_kernel_id =
        CreateKernel(program, kernel_path, all_cores, ReaderDataMovementConfig(ctime_args, kernel_defines));

    const auto cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, false);
    const CoreCoord core = cores.at(0);
    SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), dst_buffer->address()});

    return {std::move(program), {.reader_kernel_id = reader_kernel_id, .core = core}};
}

void ArgMaxSingleCoreTileFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id, shared_vars.core);
    runtime_args[0] = src_buffer->address();
    runtime_args[1] = dst_buffer->address();
}

// =============================================================================
// ArgMaxMultiCoreRowMajorFactory
// =============================================================================

ArgMaxMultiCoreRowMajorFactory::cached_program_t ArgMaxMultiCoreRowMajorFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    Program program{};

    const auto& input = tensor_args.input;

    const auto input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto input_unit_size = input.element_size();
    const auto output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const auto output_unit_size = output.element_size();

    const auto& input_shape = input.padded_shape();
    const auto rank = input_shape.size();
    const bool reduce_all = !args.dim.has_value();

    const auto red_dim_units = input_shape[rank - 1];
    const auto output_last_dim = reduce_all || args.keepdim || (rank < 2) ? 1 : input_shape[rank - 2];

    const IDevice* device = output.device();

    const auto src_buffer = input.buffer();
    const auto dst_buffer = output.buffer();
    const auto src_is_dram = src_buffer->buffer_type() == BufferType::DRAM;

    const auto alignment = src_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const auto min_red_dim_units_per_core = alignment / sizeof(bfloat16);

    auto [all_cores, cores0, cores1, red_dim_units0, red_dim_units1] =
        distribute_work_to_cores(device, red_dim_units, min_red_dim_units_per_core, args.sub_core_grids);

    const uint32_t num_cores0 = cores0.num_cores();
    const uint32_t num_cores1 = cores1.num_cores();
    const uint32_t num_total_cores = num_cores0 + num_cores1;

    const auto src_page_size = round_up_to_mul32(red_dim_units * input_unit_size);
    const auto dst_page_size = round_up_to_mul32(output_last_dim * output_unit_size);

    // Create input CB for cores0
    const uint32_t src_cb_idx = tt::CBIndex::c_0;
    const auto src_cb_page_size0 = round_up_to_mul32(red_dim_units0 * input_unit_size);
    const auto src_cb_config0 = CircularBufferConfig(src_cb_page_size0, {{src_cb_idx, input_cb_data_format}})
                                    .set_page_size(src_cb_idx, src_cb_page_size0);
    CreateCircularBuffer(program, cores0, src_cb_config0);

    if (num_cores1 > 0) {
        const auto src_cb_page_size1 = round_up_to_mul32(red_dim_units1 * input_unit_size);
        const auto src_cb_config1 = CircularBufferConfig(src_cb_page_size1, {{src_cb_idx, input_cb_data_format}})
                                        .set_page_size(src_cb_idx, src_cb_page_size1);
        CreateCircularBuffer(program, cores1, src_cb_config1);
    }

    // Create output CB
    const uint32_t dst_cb_idx = tt::CBIndex::c_1;
    const auto dst_cb_config = CircularBufferConfig(dst_page_size, {{dst_cb_idx, output_cb_data_format}})
                                   .set_page_size(dst_cb_idx, dst_page_size);
    CreateCircularBuffer(program, all_cores, dst_cb_config);

    // Create intermediate CB for indices
    const uint32_t red_idxs_cb_idx = tt::CBIndex::c_2;
    const auto red_idxs_page_size = round_up_to_mul32(output_last_dim * output_unit_size) * num_total_cores;
    const auto red_idxs_cb_config = CircularBufferConfig(red_idxs_page_size, {{red_idxs_cb_idx, output_cb_data_format}})
                                        .set_page_size(red_idxs_cb_idx, red_idxs_page_size);
    CreateCircularBuffer(program, all_cores, red_idxs_cb_config);

    // Create intermediate CB for values
    const uint32_t red_vals_cb_idx = tt::CBIndex::c_3;
    const auto red_vals_page_size = round_up_to_mul32(output_last_dim * input_unit_size) * num_total_cores;
    const auto red_vals_cb_config = CircularBufferConfig(red_vals_page_size, {{red_vals_cb_idx, input_cb_data_format}})
                                        .set_page_size(red_vals_cb_idx, red_vals_page_size);
    CreateCircularBuffer(program, all_cores, red_vals_cb_config);

    const auto inner_dim_units = output_last_dim;
    const auto outer_dim_units = input.logical_volume() / inner_dim_units / red_dim_units;

    const uint32_t reduce_core_id = 0;
    const auto cores = corerange_to_cores(all_cores, num_total_cores, true);
    const auto reduce_core = device->worker_core_from_logical_core(cores.at(reduce_core_id));

    const auto group0 = all_cores.ranges().at(0);
    const auto group1 = all_cores.size() > 1 ? all_cores.ranges().at(1) : CoreRange(CoreCoord(0, 0), CoreCoord(0, 0));

    const auto start_core0 = device->worker_core_from_logical_core(group0.start_coord);
    const auto end_core0 = device->worker_core_from_logical_core(group0.end_coord);
    const auto start_core1 = device->worker_core_from_logical_core(group1.start_coord);
    const auto end_core1 = device->worker_core_from_logical_core(group1.end_coord);

    const auto num_cores_range0 = group0.size();
    const auto num_cores_range1 = all_cores.size() > 1 ? group1.size() : 0;

    const auto start_sem_idx = CreateSemaphore(program, all_cores, 0);
    const auto done_sem_idx = CreateSemaphore(program, all_cores, 0);

    const auto src_read_size0 = red_dim_units0 * input_unit_size;
    const auto src_read_size1 = red_dim_units1 * input_unit_size;

    const int ideal_red_dim_units = (num_cores0 * red_dim_units0) + (num_cores1 * red_dim_units1);

    uint32_t red_dim_units_last0, red_dim_units_last1;
    if (num_cores1 > 0) {
        red_dim_units_last0 = red_dim_units0;
        red_dim_units_last1 = ideal_red_dim_units == red_dim_units
                                  ? red_dim_units1
                                  : red_dim_units1 - (ideal_red_dim_units - red_dim_units);
    } else {
        red_dim_units_last0 = ideal_red_dim_units == red_dim_units
                                  ? red_dim_units0
                                  : red_dim_units0 - (ideal_red_dim_units - red_dim_units);
        red_dim_units_last1 = 0;
    }

    const auto src_read_size_last0 = red_dim_units_last0 * input_unit_size;
    const auto src_read_size_last1 = red_dim_units_last1 * input_unit_size;

    std::vector<uint32_t> reader_compile_args = {
        src_cb_idx,
        dst_cb_idx,
        red_idxs_cb_idx,
        red_vals_cb_idx,
        src_page_size,
        dst_page_size,
        red_idxs_page_size / num_total_cores,
        red_vals_page_size / num_total_cores,
        outer_dim_units,
        inner_dim_units,
        red_dim_units,
        static_cast<uint32_t>(reduce_all),
        num_total_cores,
        reduce_core_id,
        static_cast<uint32_t>(reduce_core.x),
        static_cast<uint32_t>(reduce_core.y),
        static_cast<uint32_t>(end_core0.x),
        static_cast<uint32_t>(end_core0.y),
        static_cast<uint32_t>(start_core0.x),
        static_cast<uint32_t>(start_core0.y),
        static_cast<uint32_t>(end_core1.x),
        static_cast<uint32_t>(end_core1.y),
        static_cast<uint32_t>(start_core1.x),
        static_cast<uint32_t>(start_core1.y),
        static_cast<uint32_t>(num_cores_range0),
        static_cast<uint32_t>(num_cores_range1),
        start_sem_idx,
        done_sem_idx,
    };
    TensorAccessorArgs(src_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(dst_buffer).append_to(reader_compile_args);

    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp";

    KernelHandle reader_kernel_id0 = CreateKernel(
        program,
        kernel_path,
        cores0,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    KernelHandle reader_kernel_id1 = 0;
    if (num_cores1 > 0) {
        reader_kernel_id1 = CreateKernel(
            program,
            kernel_path,
            cores1,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_args});
    }

    const auto cores_coords0 = corerange_to_cores(cores0, num_cores0, true);
    const auto cores_coords1 = corerange_to_cores(cores1, num_cores1, true);

    for (uint32_t i = 0; i < num_cores0; ++i) {
        const CoreCoord& core = cores_coords0.at(i);
        SetRuntimeArgs(
            program,
            reader_kernel_id0,
            core,
            {src_buffer->address(),
             dst_buffer->address(),
             i,
             i * src_read_size0,
             i * red_dim_units0,
             (i == num_cores0 - 1) ? src_read_size_last0 : src_read_size0,
             (i == num_cores0 - 1) ? red_dim_units_last0 : red_dim_units0});
    }

    const uint32_t src_offset1 = static_cast<uint32_t>(src_read_size0 * num_cores0);
    const uint32_t red_dim_offset1 = static_cast<uint32_t>(red_dim_units0 * num_cores0);

    for (uint32_t i = 0; i < num_cores1; ++i) {
        const CoreCoord& core = cores_coords1.at(i);
        SetRuntimeArgs(
            program,
            reader_kernel_id1,
            core,
            {src_buffer->address(),
             dst_buffer->address(),
             static_cast<uint32_t>(num_cores0 + i),
             src_offset1 + (i * src_read_size1),
             red_dim_offset1 + (i * red_dim_units1),
             (i == num_cores1 - 1) ? src_read_size_last1 : src_read_size1,
             (i == num_cores1 - 1) ? red_dim_units_last1 : red_dim_units1});
    }

    return {
        std::move(program),
        {.reader_kernel_id0 = reader_kernel_id0,
         .reader_kernel_id1 = reader_kernel_id1,
         .cores_coords0 = cores_coords0,
         .cores_coords1 = cores_coords1,
         .num_cores0 = num_cores0,
         .num_cores1 = num_cores1}};
}

void ArgMaxMultiCoreRowMajorFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    for (const auto& core : shared_vars.cores_coords0) {
        auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id0, core);
        runtime_args[0] = src_buffer->address();
        runtime_args[1] = dst_buffer->address();
    }

    for (const auto& core : shared_vars.cores_coords1) {
        auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id1, core);
        runtime_args[0] = src_buffer->address();
        runtime_args[1] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::reduction::argmax::program
