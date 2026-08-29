// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_device_operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

#include <utility>

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

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
    uint32_t red_dim_units0 = 0, red_dim_units1 = 0;

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
 * Dataflow Buffers (DFBs):
 * 1. Input DFB:
 *    - Size depends on input tensor shape and number of cores
 *    - Used for reading input tensor data
 *
 * 2. Worker Output DFB (indices):
 *    - Size depends on final output shape and number of worker cores
 *    - Used by worker cores to store local maxima indices
 *
 * 3. Worker Output DFB (values):
 *    - Size depends on final output shape and number of worker cores
 *    - Used by worker cores to store local maxima values
 *
 * 4. Final Output DFB:
 *    - Size depends on final output tensor shape
 *    - Used only by reduce core to write final global argmax results
 *
 * Core Roles:
 * 1. Worker Cores:
 *    - Process assigned portion of reduction dimension
 *    - Find local maxima and their indices
 *    - Write results to output DFB
 *
 * 2. Reduce Core:
 *    - Collects results from all worker cores
 *    - Performs final reduction to find global maxima
 *    - Writes final results to DRAM
 *
 * Semaphore Usage:
 * 1. Semaphore 1 (start):
 *    - Controls output buffer availability for writing results
 *    - Worker cores wait before writing results
 *    - Set by reduce core (multicast)
 *
 * 2. Semaphore 2 (done):
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
ttnn::device_operation::ProgramArtifacts ArgMaxMultiCoreProgramFactory::create_program_artifacts(
    const ArgmaxParams& operation_attributes, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const auto& dim = operation_attributes.dim;
    const bool keepdim = operation_attributes.keepdim;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    // Resource names. Declared function-locally: both argmax factories share a unity-build
    // translation unit, and namespace-scope `const` objects would collide by name there.
    const KernelSpecName READER0{"reader0"};
    const KernelSpecName READER1{"reader1"};
    const DFBSpecName SRC0{"src0"};
    const DFBSpecName SRC1{"src1"};
    const DFBSpecName DST{"dst"};
    const DFBSpecName RED_IDXS{"red_idxs"};
    const DFBSpecName RED_VALS{"red_vals"};
    const SemaphoreSpecName START{"start"};
    const SemaphoreSpecName DONE{"done"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    const auto input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto input_unit_size = input.element_size();
    const auto output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const auto output_unit_size = output.element_size();

    const auto& input_shape = input.padded_shape();
    const auto rank = input_shape.size();
    const bool reduce_all = not dim.has_value();

    // Last dimension in input i.e. reduction dimension
    const auto red_dim_units = input_shape[rank - 1];

    // Last dimension in output i.e. the dim left after reduction
    const auto output_last_dim = reduce_all or keepdim or (rank < 2) ? 1 : input_shape[rank - 2];

    const tt::tt_metal::IDevice* device = &output.mutable_device();

    const auto src_is_dram = input.mesh_buffer().device_local_config().buffer_type == tt::tt_metal::BufferType::DRAM;

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

    TT_FATAL(num_total_cores > 0, "Argmax multicore requires at least one worker core");
    validate_reduce_op_program_grid(
        "Argmax multicore",
        all_cores,
        device->compute_with_storage_grid_size(),
        sub_core_grids.has_value() ? &sub_core_grids.value() : nullptr,
        true,
        {});

    // Page sizes for input and output tensors based on the ROW_MAJOR layout
    const auto src_page_size = round_up_to_mul32(red_dim_units * input_unit_size);
    const auto dst_page_size = round_up_to_mul32(output_last_dim * output_unit_size);

    // DFBs are declared in the order the pre-Metal-2.0 factory declared its circular buffers. The
    // allocator walks this list in order and gives each DFB one L1 address across every node it
    // spans, so preserving the order preserves the resulting addresses.
    Group<DataflowBufferSpec> dataflow_buffers;

    // Input DFB to read reduction dim worth of data at once (split across all cores). The two core
    // groups get different per-core block counts, so group 1 needs its own (differently sized) DFB.
    const auto src_dfb_entry_size0 = round_up_to_mul32(red_dim_units0 * input_unit_size);
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC0,
        .entry_size = src_dfb_entry_size0,
        .num_entries = 1,
        .data_format_metadata = input_dfb_data_format,
    });

    // We only create the second input DFB if there are some cores assigned to the second group
    if (num_cores1 > 0) {
        const auto src_dfb_entry_size1 = round_up_to_mul32(red_dim_units1 * input_unit_size);
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SRC1,
            .entry_size = src_dfb_entry_size1,
            .num_entries = 1,
            .data_format_metadata = input_dfb_data_format,
        });
    }

    // Create output DFB based on the output shape's last dimension
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DST,
        .entry_size = dst_page_size,
        .num_entries = 1,
        .data_format_metadata = output_dfb_data_format,
    });

    // Create intermediate DFB for indices based on number of cores and output shape's last dimension
    const auto red_idxs_page_size = round_up_to_mul32(output_last_dim * output_unit_size) * num_total_cores;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RED_IDXS,
        .entry_size = red_idxs_page_size,
        .num_entries = 1,
        .data_format_metadata = output_dfb_data_format,
    });

    // Create intermediate DFB for values based on number of cores and output shape's last dimension
    const auto red_vals_page_size = round_up_to_mul32(output_last_dim * input_unit_size) * num_total_cores;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RED_VALS,
        .entry_size = red_vals_page_size,
        .num_entries = 1,
        .data_format_metadata = input_dfb_data_format,
    });

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

    // Two semaphores for synchronization (cores -> reducer core) and (reducer core -> cores).
    // Semaphores are zero-initialized, which is the initial value the kernel handshake assumes.
    SemaphoreSpec start_sem{.unique_id = START, .target_nodes = all_cores};
    SemaphoreSpec done_sem{.unique_id = DONE, .target_nodes = all_cores};

    // Byte size of the data to read from the input DFB for each core
    const auto src_read_size0 = red_dim_units0 * input_unit_size;
    const auto src_read_size1 = red_dim_units1 * input_unit_size;

    // If red_dim_units is not a multiple of min_red_dim_units_per_core, then the last core will read a smaller amount
    // of data We calculate that number here
    const int ideal_red_dim_units = (num_cores0 * red_dim_units0) + (num_cores1 * red_dim_units1);

    uint32_t red_dim_units_last0 = 0, red_dim_units_last1 = 0;
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

    // Common compile time args for all cores.
    // Names are the reader kernel's own variable names; refer to the kernel code for what each means.
    //
    // start_core_*/end_core_* carry the NOC1 multicast convention: a NOC1 multicast rectangle is
    // addressed end-corner first, so the kernel's "start" arguments receive the group's *end*
    // coordinate and its "end" arguments receive the *start* one. The swap is deliberate; the
    // kernel feeds these straight to set_multicast().
    const KernelSpec::CompileTimeArgs reader_compile_args = {
        // The reader sizes its transfers from the src_read_size runtime argument, so it never reads
        // src_page_size. Emitted anyway, unchanged from the pre-Metal-2.0 argument list.
        {"src_page_size", src_page_size},
        {"dst_page_size", dst_page_size},
        {"red_idx_size_per_core", red_idxs_page_size / num_total_cores},
        {"red_val_size_per_core", red_vals_page_size / num_total_cores},
        {"outer_dim_units", outer_dim_units},
        {"inner_dim_units", inner_dim_units},
        {"red_dim_units", red_dim_units},
        {"reduce_all", static_cast<uint32_t>(reduce_all)},
        {"num_cores", num_total_cores},
        {"reduce_core_id", reduce_core_id},
        {"reduce_core_x", static_cast<uint32_t>(reduce_core.x)},
        {"reduce_core_y", static_cast<uint32_t>(reduce_core.y)},
        {"start_core_x0", static_cast<uint32_t>(end_core0.x)},
        {"start_core_y0", static_cast<uint32_t>(end_core0.y)},
        {"end_core_x0", static_cast<uint32_t>(start_core0.x)},
        {"end_core_y0", static_cast<uint32_t>(start_core0.y)},
        {"start_core_x1", static_cast<uint32_t>(end_core1.x)},
        {"start_core_y1", static_cast<uint32_t>(end_core1.y)},
        {"end_core_x1", static_cast<uint32_t>(start_core1.x)},
        {"end_core_y1", static_cast<uint32_t>(start_core1.y)},
        {"num_cores0", static_cast<uint32_t>(num_cores_range0)},
        {"num_cores1", static_cast<uint32_t>(num_cores_range1)},
    };

    const std::filesystem::path reader_source =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp";

    // Every DFB this reader binds is touched by the reader alone, through a raw write pointer, with
    // no FIFO operation anywhere in the kernel — so each is bound self-loop (the one toucher is both
    // producer and consumer). The two readers cover disjoint node sets, so a DFB they both bind
    // still sees exactly one instance of each endpoint per node.
    //
    // The cross-core writes into the reducer's L1 do not add an endpoint: their destination is a
    // bare NoC address, not a binding.
    auto make_reader = [&](const KernelSpecName& unique_id, const DFBSpecName& src_dfb) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = reader_source,
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = src_dfb, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER},
                    DFBBinding{
                        .dfb_spec_name = src_dfb, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = DST, .accessor_name = "dst", .endpoint_type = DFBEndpointType::PRODUCER},
                    DFBBinding{
                        .dfb_spec_name = DST, .accessor_name = "dst", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = RED_IDXS,
                        .accessor_name = "red_idxs",
                        .endpoint_type = DFBEndpointType::PRODUCER},
                    DFBBinding{
                        .dfb_spec_name = RED_IDXS,
                        .accessor_name = "red_idxs",
                        .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = RED_VALS,
                        .accessor_name = "red_vals",
                        .endpoint_type = DFBEndpointType::PRODUCER},
                    DFBBinding{
                        .dfb_spec_name = RED_VALS,
                        .accessor_name = "red_vals",
                        .endpoint_type = DFBEndpointType::CONSUMER},
                },
            .semaphore_bindings =
                {
                    SemaphoreBinding{.semaphore_spec_name = START, .accessor_name = "start"},
                    SemaphoreBinding{.semaphore_spec_name = DONE, .accessor_name = "done"},
                },
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
                    TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
                },
            .compile_time_args = reader_compile_args,
            .runtime_arg_schema =
                {
                    .runtime_arg_names =
                        {"core_id", "src_offset", "red_dim_offset", "src_read_size", "red_dim_units_this_core"},
                },
            // Not the reader default placement (that is NOC_0): this kernel is pinned to RISCV_1 on
            // NOC_1, reproduced field-by-field from the pre-Metal-2.0 config.
            .hw_config = DataMovementHardwareConfig{DataMovementGen1Config{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_1,
                .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            }},
        };
    };

    Group<KernelSpec> kernels;
    Group<WorkUnitSpec> work_units;
    Group<KernelRunArgs> kernel_run_args;

    kernels.push_back(make_reader(READER0, SRC0));
    work_units.push_back(WorkUnitSpec{.name = "group0", .kernels = {READER0}, .target_nodes = cores0});

    const auto cores_coords0 = corerange_to_cores(cores0, num_cores0, true);
    const auto cores_coords1 = corerange_to_cores(cores1, num_cores1, true);

    // Set runtime args for cores0 and cores1, only offsets (src and red_dim_units) are different
    // Refer to the kernel code for explanation of the args
    KernelRunArgs reader_run_args0{.kernel = READER0};
    for (uint32_t i = 0; i < num_cores0; ++i) {
        const CoreCoord& core = cores_coords0.at(i);
        AddRuntimeArgsForNode(
            reader_run_args0.runtime_arg_values,
            core,
            {{"core_id", i},
             {"src_offset", static_cast<uint32_t>(i * src_read_size0)},
             {"red_dim_offset", i * red_dim_units0},
             {"src_read_size", static_cast<uint32_t>((i == num_cores0 - 1) ? src_read_size_last0 : src_read_size0)},
             {"red_dim_units_this_core", (i == num_cores0 - 1) ? red_dim_units_last0 : red_dim_units0}});
    }
    kernel_run_args.push_back(std::move(reader_run_args0));

    if (num_cores1 > 0) {
        kernels.push_back(make_reader(READER1, SRC1));
        work_units.push_back(WorkUnitSpec{.name = "group1", .kernels = {READER1}, .target_nodes = cores1});

        const uint32_t src_offset1 = static_cast<uint32_t>(src_read_size0 * num_cores0);
        const uint32_t red_dim_offset1 = static_cast<uint32_t>(red_dim_units0 * num_cores0);

        KernelRunArgs reader_run_args1{.kernel = READER1};
        for (uint32_t i = 0; i < num_cores1; ++i) {
            const CoreCoord& core = cores_coords1.at(i);
            AddRuntimeArgsForNode(
                reader_run_args1.runtime_arg_values,
                core,
                {{"core_id", static_cast<uint32_t>(num_cores0 + i)},
                 {"src_offset", static_cast<uint32_t>(src_offset1 + (i * src_read_size1))},
                 {"red_dim_offset", red_dim_offset1 + (i * red_dim_units1)},
                 {"src_read_size", static_cast<uint32_t>((i == num_cores1 - 1) ? src_read_size_last1 : src_read_size1)},
                 {"red_dim_units_this_core", (i == num_cores1 - 1) ? red_dim_units_last1 : red_dim_units1}});
        }
        kernel_run_args.push_back(std::move(reader_run_args1));
    }

    ProgramSpec spec{
        .name = "argmax_multi_core",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = {std::move(start_sem), std::move(done_sem)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = std::move(kernel_run_args);
    run_args.tensor_args = {
        {INPUT, input},
        {OUTPUT, output},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
