// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_multi_core_program_factory.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::reduction::argmax::program {

using namespace tt::tt_metal;

/**
 * @brief Distributes work across cores for argmax reduction operations
 *
 * If a sub_core_grids is provided, it will be used to distribute the work evenly across the cores.
 * Otherwise, we distribute to maximum of two core groups, with each core group getting a minimum of
 * `min_red_dim_units_per_core` elements to process, except the last core.
 * @param device Pointer to the device
 * @param red_dim_units Total units in the reduction dimension
 * @param min_red_dim_units_per_core Minimum units per core (for alignment)
 * @param sub_core_grids Optional core grid specification
 * @return Tuple containing distribution parameters:
 *         - all_cores: CoreRangeSet of all cores
 *         - cores0: First group of cores
 *         - cores1: Second group of cores (if any)
 *         - red_dim_units0: Units assigned to first group per core
 *         - red_dim_units1: Units assigned to second group per core
 */
static inline std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> distribute_work_to_cores(
    const tt::tt_metal::IDevice* device,
    const uint32_t red_dim_units,
    const uint32_t min_red_dim_units_per_core,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    CoreRangeSet all_cores, cores0, cores1;
    uint32_t red_dim_units0, red_dim_units1;

    if (sub_core_grids.has_value()) {
        all_cores = sub_core_grids.value();
        // If there are two core groups, assign to cores0 and cores1
        // Otherwise, assign to cores0
        // Ensure red_dim is divided in blocks of min_red_dim_units_per_core
        const uint32_t total_blocks = tt::div_up(red_dim_units, min_red_dim_units_per_core);

        if (all_cores.size() == 2) {
            cores0 = CoreRangeSet(all_cores.ranges().at(0));
            cores1 = CoreRangeSet(all_cores.ranges().at(1));

            // Ensure red_dim is divided in blocks of min_red_dim_units_per_core, equally to all cores
            const uint32_t total_cores = cores0.num_cores() + cores1.num_cores();
            const uint32_t blocks_per_core = tt::div_up(total_blocks, total_cores);

            red_dim_units0 = blocks_per_core * min_red_dim_units_per_core;
            red_dim_units1 = blocks_per_core * min_red_dim_units_per_core;
            ;
        } else {
            // If there is only one core group, assign to cores0, keep cores1 empty
            cores0 = all_cores;
            cores1 = CoreRangeSet();

            const auto total_cores = cores0.num_cores();
            const uint32_t blocks_per_core = tt::div_up(total_blocks, total_cores);

            red_dim_units0 = blocks_per_core * min_red_dim_units_per_core;
            red_dim_units1 = 0;
        }
    } else {
        // We pick as many cores as possible, but each core will read a multiple of min_red_dim_units_per_core
        const auto core_grid = device->compute_with_storage_grid_size();
        uint32_t num_total_cores;
        std::tie(num_total_cores, all_cores, cores0, cores1, red_dim_units0, red_dim_units1) =
            tt::tt_metal::split_work_to_cores(core_grid, tt::div_up(red_dim_units, min_red_dim_units_per_core));
        red_dim_units0 *= min_red_dim_units_per_core;
        red_dim_units1 *= min_red_dim_units_per_core;
    }

    return {all_cores, cores0, cores1, red_dim_units0, red_dim_units1};
}

/*
 * Design of argmax_multi_core:
 *
 * The argmax operation is split across multiple cores to handle large tensors efficiently.
 * Each core processes a portion of the reduction dimension, finding local maxima and their indices.
 *
 * Circular Buffers (CBs):
 * 1. Input CB:
 *    - Size depends on input tensor shape and number of cores
 *    - Used for reading input tensor data
 *
 * 2. Worker Output CB (indices):
 *    - Size depends on final output shape and number of worker cores
 *    - Used by worker cores to store local maxima indices
 *
 * 3. Worker Output CB (values):
 *    - Size depends on final output shape and number of worker cores
 *    - Used by worker cores to store local maxima values
 *
 * 4. Final Output CB:
 *    - Size depends on final output tensor shape
 *    - Used only by reduce core to write final global argmax results
 *
 * Core Roles:
 * 1. Worker Cores:
 *    - Process assigned portion of reduction dimension
 *    - Find local maxima and their indices
 *    - Write results to output CB
 *
 * 2. Reduce Core:
 *    - Collects results from all worker cores
 *    - Performs final reduction to find global maxima
 *    - Writes final results to DRAM
 *
 * Semaphore Usage:
 * 1. Semaphore 1:
 *    - Controls output buffer availability for writing results
 *    - Worker cores wait before writing results
 *    - Set by reduce core (multicast)
 *
 * 2. Semaphore 2:
 *    - Controls output buffer availability for reading results
 *    - Worker cores signal completion (increment)
 *    - Reduce core waits for all workers
 *
 * Multicast Design:
 *
 * 1. Core Groups:
 *    - Cores are split into two groups (cores0 and cores1) for balanced workload
 *    - Each group handles a different portion of the reduction dimension
 *    - cores0 handles red_dim_units0 elements
 *    - cores1 handles red_dim_units1 elements
 *    - Each core gets a minimum of `min_red_dim_units_per_core` elements to process, except the last core
 *
 * 2. Core Layout:
 *    - Cores are arranged in a grid pattern
 *    - Example for 4x4 grid:
 *
 *      +---+---+---+---+
 *      |R0 |W1 |W2 |W3 |
 *      +---+---+---+---+
 *      |W4 |W5 |W6 |W7 |
 *      +---+---+---+---+
 *      |W8 |W9 |W10|W11|
 *      +---+---+---+---+
 *      |W12|W13|W14|W15|
 *      +---+---+---+---+
 *
 *    Where R0 is reduce core, W* are worker cores
 *    There may be two grids (based on the number of cores)
 *
 *    Refer to the kernel code for info on compile time args and runtime args
 */
ArgMaxMultiCoreProgramFactory::cached_program_t ArgMaxMultiCoreProgramFactory::create(
    const ArgmaxParams& operation_attributes, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& dim = operation_attributes.dim;
    const bool keepdim = operation_attributes.keepdim;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::tt_metal::Program program{};

    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto input_unit_size = input.element_size();
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const auto output_unit_size = output.element_size();

    const auto& input_shape = input.padded_shape();
    const auto rank = input_shape.size();
    const bool reduce_all = not dim.has_value();

    // Last dimension in input i.e. reduction dimension
    const auto red_dim_units = input_shape[rank - 1];

    // Last dimension in output i.e. the dim left after reduction
    const auto output_last_dim = reduce_all or keepdim or (rank < 2) ? 1 : input_shape[rank - 2];

    const tt::tt_metal::IDevice* device = output.device();

    auto* const src_buffer = input.buffer();
    auto* const dst_buffer = output.buffer();
    const auto src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // NOC transactions need to be aligned.
    // So, for bfloat16 dtype, we need at least 16/32 units per core (depending on alignment) to avoid unaligned
    // accesses.
    const auto alignment = src_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const auto min_red_dim_units_per_core = alignment / sizeof(bfloat16);

    // Distribute work to cores
    auto [all_cores, cores0, cores1, red_dim_units0, red_dim_units1] =
        distribute_work_to_cores(device, red_dim_units, min_red_dim_units_per_core, sub_core_grids);

    const uint32_t num_cores0 = cores0.num_cores();
    const uint32_t num_cores1 = cores1.num_cores();
    const uint32_t num_total_cores = num_cores0 + num_cores1;

    // Page sizes for input and output tensors based on the ROW_MAJOR layout
    const auto src_page_size = round_up_to_mul32(red_dim_units * input_unit_size);
    const auto dst_page_size = round_up_to_mul32(output_last_dim * output_unit_size);

    // Create input CB to read reduction dim worth of data at once (split across all cores)
    const uint32_t src_cb_idx = tt::CBIndex::c_0;
    const auto src_cb_page_size0 = round_up_to_mul32(red_dim_units0 * input_unit_size);
    const auto src_cb_config0 =
        tt::tt_metal::CircularBufferConfig(src_cb_page_size0, {{src_cb_idx, input_cb_data_format}})
            .set_page_size(src_cb_idx, src_cb_page_size0);
    tt::tt_metal::CreateCircularBuffer(program, cores0, src_cb_config0);

    // We only create the second CB if there are some cores assigned to the second group
    if (num_cores1 > 0) {
        const auto src_cb_page_size1 = round_up_to_mul32(red_dim_units1 * input_unit_size);
        const auto src_cb_config1 =
            tt::tt_metal::CircularBufferConfig(src_cb_page_size1, {{src_cb_idx, input_cb_data_format}})
                .set_page_size(src_cb_idx, src_cb_page_size1);
        tt::tt_metal::CreateCircularBuffer(program, cores1, src_cb_config1);
    }

    // Create output CB based on the output shape's last dimension
    const uint32_t dst_cb_idx = tt::CBIndex::c_1;
    const auto dst_db_config = tt::tt_metal::CircularBufferConfig(dst_page_size, {{dst_cb_idx, output_cb_data_format}})
                                   .set_page_size(dst_cb_idx, dst_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, dst_db_config);

    // Create intermediate CB for indices based on number of cores and output shape's last dimension
    const uint32_t red_idxs_cb_idx = tt::CBIndex::c_2;
    const auto red_idxs_page_size = round_up_to_mul32(output_last_dim * output_unit_size) * num_total_cores;
    const auto red_idxs_db_config =
        tt::tt_metal::CircularBufferConfig(red_idxs_page_size, {{red_idxs_cb_idx, output_cb_data_format}})
            .set_page_size(red_idxs_cb_idx, red_idxs_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, red_idxs_db_config);

    // Create intermediate CB for values based on number of cores and output shape's last dimension
    const uint32_t red_vals_cb_idx = tt::CBIndex::c_3;
    const auto red_vals_page_size = round_up_to_mul32(output_last_dim * input_unit_size) * num_total_cores;
    const auto red_vals_cb_config =
        tt::tt_metal::CircularBufferConfig(red_vals_page_size, {{red_vals_cb_idx, input_cb_data_format}})
            .set_page_size(red_vals_cb_idx, red_vals_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, red_vals_cb_config);

    const auto inner_dim_units = output_last_dim;
    const auto outer_dim_units = input.logical_volume() / inner_dim_units / red_dim_units;

    // Get physical coordinates of the reduce core that collates the intermediate outputs
    const uint32_t reduce_core_id = 0;  // We can do perf optimization by tuning this in the future
    const auto cores = corerange_to_cores(all_cores, num_total_cores, true);
    const auto reduce_core = device->worker_core_from_logical_core(cores.at(reduce_core_id));

    // Get first and last core's coordinates for the at max two groups of cores in all_cores
    const auto group0 = all_cores.ranges().at(0);
    const auto group1 = all_cores.size() > 1 ? all_cores.ranges().at(1) : CoreRange(CoreCoord(0, 0), CoreCoord(0, 0));

    const auto start_core0 = device->worker_core_from_logical_core(group0.start_coord);
    const auto end_core0 = device->worker_core_from_logical_core(group0.end_coord);
    const auto start_core1 = device->worker_core_from_logical_core(group1.start_coord);
    const auto end_core1 = device->worker_core_from_logical_core(group1.end_coord);

    const auto num_cores_range0 = group0.size();
    const auto num_cores_range1 = all_cores.size() > 1 ? group1.size() : 0;

    // Allocate two semaphores for synchronization (cores -> reducer core) and (reducer core -> cores)
    const auto start_sem_idx = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const auto done_sem_idx = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // Byte size of the data to read from the input CB for each core
    const auto src_read_size0 = red_dim_units0 * input_unit_size;
    const auto src_read_size1 = red_dim_units1 * input_unit_size;

    // If red_dim_units is not a multiple of min_red_dim_units_per_core, then the last core will read a smaller amount
    // of data We calculate that number here
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

    // Common compile time args for all cores
    // Refer to the kernel code for explanation of the args
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
        (uint32_t)(reduce_all),
        num_total_cores,
        reduce_core_id,
        (uint32_t)reduce_core.x,
        (uint32_t)reduce_core.y,
        // end comes before start for NOC1
        (uint32_t)end_core0.x,
        (uint32_t)end_core0.y,
        (uint32_t)start_core0.x,
        (uint32_t)start_core0.y,
        (uint32_t)end_core1.x,
        (uint32_t)end_core1.y,
        (uint32_t)start_core1.x,
        (uint32_t)start_core1.y,
        (uint32_t)num_cores_range0,
        (uint32_t)num_cores_range1,
        start_sem_idx,
        done_sem_idx,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(reader_compile_args);

    std::map<std::string, std::string> kernel_defines;
    tt::tt_metal::KernelHandle reader_kernel_id0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp",
        cores0,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    tt::tt_metal::KernelHandle reader_kernel_id1 = 0;
    if (num_cores1 > 0) {
        reader_kernel_id1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp",
            cores1,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = reader_compile_args});
    }
    const auto cores_coords0 = corerange_to_cores(cores0, num_cores0, true);
    const auto cores_coords1 = corerange_to_cores(cores1, num_cores1, true);

    // Set runtime args for cores0 and cores1, only offsets (src and red_dim_units) are different
    // Refer to the kernel code for explanation of the args
    for (uint32_t i = 0; i < num_cores0; ++i) {
        const CoreCoord& core = cores_coords0.at(i);
        tt::tt_metal::SetRuntimeArgs(
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
        tt::tt_metal::SetRuntimeArgs(
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

    return {std::move(program), {reader_kernel_id0, reader_kernel_id1, cores_coords0, cores_coords1}};
}

void ArgMaxMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ArgmaxParams& /*operation_attributes*/,
    const ArgmaxInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    auto& program = cached_program.program;
    const auto& reader_kernel_id0 = cached_program.shared_variables.reader_kernel_id0;
    const auto& reader_kernel_id1 = cached_program.shared_variables.reader_kernel_id1;
    const auto& cores_coords0 = cached_program.shared_variables.cores_coords0;
    const auto& cores_coords1 = cached_program.shared_variables.cores_coords1;

    for (const auto& core : cores_coords0) {
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id0, core);
        reader_runtime_args[0] = src_buffer->address();
        reader_runtime_args[1] = dst_buffer->address();
    }
    for (const auto& core : cores_coords1) {
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id1, core);
        reader_runtime_args[0] = src_buffer->address();
        reader_runtime_args[1] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::reduction::argmax::program
