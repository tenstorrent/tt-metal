// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_device_operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ttnn::prim {

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
ProgramDescriptor ArgMaxMultiCoreProgramFactory::create_descriptor(
    const ArgmaxParams& operation_attributes, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const auto& dim = operation_attributes.dim;
    const bool keepdim = operation_attributes.keepdim;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    ProgramDescriptor desc;

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

    const auto inner_dim_units = output_last_dim;
    const auto outer_dim_units = input.logical_volume() / inner_dim_units / red_dim_units;

    // Hoisted above CB creation: the L1 budget check below needs these sizes.
    const auto red_idxs_page_size = round_up_to_mul32(output_last_dim * output_unit_size) * num_total_cores;
    const auto red_vals_page_size = round_up_to_mul32(output_last_dim * input_unit_size) * num_total_cores;

    const auto src_cb_page_size0 = round_up_to_mul32(red_dim_units0 * input_unit_size);
    const auto src_cb_page_size1 = num_cores1 > 0 ? round_up_to_mul32(red_dim_units1 * input_unit_size) : 0u;

    // Doubling src_cb is the only extra L1 cost, but it has to be charged against the region the
    // CBs actually get: base allocator address up to lowest_occupied_compute_l1_address(), with
    // each CB rounded to the DRAM alignment (see ProgramImpl::allocate_circular_buffers).
    const uint32_t cb_alignment = device->allocator()->get_alignment(tt::tt_metal::BufferType::DRAM);
    const auto cb_alloc_size = [cb_alignment](uint64_t cb_size) {
        return tt::align(cb_size, static_cast<uint64_t>(cb_alignment));
    };

    // A core belongs to exactly one src_cb group, so charge the larger group.
    const uint64_t shared_cb_bytes =
        cb_alloc_size(dst_page_size) + cb_alloc_size(red_idxs_page_size) + cb_alloc_size(red_vals_page_size);
    const uint64_t dual_src_cb_bytes =
        std::max(cb_alloc_size(2ull * src_cb_page_size0), cb_alloc_size(2ull * src_cb_page_size1));

    const auto lowest_occupied_l1 = device->lowest_occupied_compute_l1_address();
    const uint64_t cb_region_end =
        lowest_occupied_l1.has_value() ? lowest_occupied_l1.value() : device->l1_size_per_core();
    const uint64_t cb_region_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint64_t l1_available_for_cbs = cb_region_end > cb_region_base ? cb_region_end - cb_region_base : 0;

    const bool l1_fits_secondary_dm = (dual_src_cb_bytes + shared_cb_bytes) <= l1_available_for_cbs;

    // Split the inner (j) loop across both data movement processors; a single kernel leaves
    // RISCV_0 idle. The halves fill disjoint entries of this core's partial slot, so the cross-core
    // protocol is unchanged. reduce_all is excluded: its accumulator spans the whole j range, so it
    // would need a value merge rather than a partition.
    const bool use_secondary_dm = (not reduce_all) && (inner_dim_units >= 2) && l1_fits_secondary_dm;
    const uint32_t src_cb_copies = use_secondary_dm ? 2 : 1;

    // Even split, biased to the secondary when odd. The split point is a compile time arg shared
    // by a whole core group, so it cannot be tuned per core.
    const uint32_t primary_j_end = use_secondary_dm ? (inner_dim_units / 2) : inner_dim_units;

    log_debug(
        tt::LogOp,
        "argmax multicore: cores={} inner_dim_units={} red_dim_units={} secondary_dm={} primary_j_end={}",
        num_total_cores,
        inner_dim_units,
        red_dim_units,
        use_secondary_dm,
        primary_j_end);

    // Create input CB to read reduction dim worth of data at once (split across all cores)
    const uint32_t src_cb_idx = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = src_cb_copies * src_cb_page_size0,
        .core_ranges = cores0,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src_cb_idx),
            .data_format = input_cb_data_format,
            .page_size = src_cb_page_size0,
        }}},
    });

    // We only create the second CB if there are some cores assigned to the second group
    if (num_cores1 > 0) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = src_cb_copies * src_cb_page_size1,
            .core_ranges = cores1,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src_cb_idx),
                .data_format = input_cb_data_format,
                .page_size = src_cb_page_size1,
            }}},
        });
    }

    // Create output CB based on the output shape's last dimension
    const uint32_t dst_cb_idx = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = dst_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(dst_cb_idx),
            .data_format = output_cb_data_format,
            .page_size = dst_page_size,
        }}},
    });

    // Create intermediate CB for indices based on number of cores and output shape's last dimension
    const uint32_t red_idxs_cb_idx = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = red_idxs_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(red_idxs_cb_idx),
            .data_format = output_cb_data_format,
            .page_size = red_idxs_page_size,
        }}},
    });

    // Create intermediate CB for values based on number of cores and output shape's last dimension
    const uint32_t red_vals_cb_idx = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = red_vals_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(red_vals_cb_idx),
            .data_format = input_cb_data_format,
            .page_size = red_vals_page_size,
        }}},
    });

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
    const uint32_t start_sem_idx = 0;
    const uint32_t done_sem_idx = 1;
    // Secondary -> primary handoff, local to one core. Always allocated so semaphore ids do not
    // depend on use_secondary_dm.
    const uint32_t partial_ready_sem_idx = 2;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = start_sem_idx,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = done_sem_idx,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = partial_ready_sem_idx,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

    // Byte size of the data to read from the input CB for each core
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

    // Common compile time args for all cores and both data movement processors
    // Refer to the kernel code for explanation of the args
    const std::vector<uint32_t> base_compile_args = {
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
        // end comes before start for NOC1
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

    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp";

    // Per-instance args: this processor's j sub-range, its half of src_cb, and whether it owns the
    // cross-core reduction.
    const auto make_compile_args = [&](uint32_t j_start, uint32_t j_end, uint32_t src_cb_offset, bool owns_reduction) {
        std::vector<uint32_t> args = base_compile_args;
        args.push_back(partial_ready_sem_idx);
        args.push_back(j_start);
        args.push_back(j_end);
        args.push_back(src_cb_offset);
        args.push_back(static_cast<uint32_t>(owns_reduction));
        args.push_back(static_cast<uint32_t>(use_secondary_dm));
        tt::tt_metal::TensorAccessorArgs(input).append_to(args);
        tt::tt_metal::TensorAccessorArgs(output).append_to(args);
        return args;
    };

    const auto make_kernel = [&](const CoreRangeSet& kernel_cores,
                                 tt::tt_metal::DataMovementProcessor processor,
                                 tt::tt_metal::NOC noc,
                                 std::vector<uint32_t> args) {
        KernelDescriptor k;
        k.kernel_source = kernel_path;
        k.source_type = KernelDescriptor::SourceType::FILE_PATH;
        k.core_ranges = kernel_cores;
        k.compile_time_args = std::move(args);
        k.config = DataMovementConfigDescriptor{.processor = processor, .noc = noc};
        return k;
    };

    // RISCV_1 owns the cross-core protocol (start_sem multicast, done_sem, partial-slot write,
    // final reduction); its NOC1 pinning is asserted in the kernel. RISCV_0 only scans its j
    // sub-range and hands off through partial_ready_sem.
    const auto make_primary = [&](const CoreRangeSet& kernel_cores) {
        return make_kernel(
            kernel_cores,
            tt::tt_metal::DataMovementProcessor::RISCV_1,
            tt::tt_metal::NOC::RISCV_1_default,
            make_compile_args(0, primary_j_end, 0, /*owns_reduction=*/true));
    };
    const auto make_secondary = [&](const CoreRangeSet& kernel_cores, uint32_t src_cb_offset) {
        return make_kernel(
            kernel_cores,
            tt::tt_metal::DataMovementProcessor::RISCV_0,
            tt::tt_metal::NOC::RISCV_0_default,
            make_compile_args(primary_j_end, inner_dim_units, src_cb_offset, /*owns_reduction=*/false));
    };

    // Both processors on a core take identical runtime args; only their compile time args differ.
    const auto emplace_core_args = [&](KernelDescriptor& kernel,
                                       const CoreCoord& core,
                                       uint32_t core_index,
                                       uint32_t src_offset,
                                       uint32_t red_dim_offset,
                                       uint32_t read_size,
                                       uint32_t units_this_core) {
        kernel.emplace_runtime_args(
            core, {input, output, core_index, src_offset, red_dim_offset, read_size, units_this_core});
    };

    KernelDescriptor reader_desc0 = make_primary(cores0);
    std::optional<KernelDescriptor> secondary_desc0;
    if (use_secondary_dm) {
        secondary_desc0 = make_secondary(cores0, src_cb_page_size0);
    }

    const auto cores_coords0 = corerange_to_cores(cores0, num_cores0, true);
    const auto cores_coords1 = corerange_to_cores(cores1, num_cores1, true);

    // Set runtime args for cores0 and cores1, only offsets (src and red_dim_units) are different
    // Refer to the kernel code for explanation of the args.
    for (uint32_t i = 0; i < num_cores0; ++i) {
        const CoreCoord& core = cores_coords0.at(i);
        const bool is_last = (i == num_cores0 - 1);
        const uint32_t read_size = static_cast<uint32_t>(is_last ? src_read_size_last0 : src_read_size0);
        const uint32_t units = is_last ? red_dim_units_last0 : red_dim_units0;
        const uint32_t src_offset = static_cast<uint32_t>(i * src_read_size0);
        const uint32_t red_dim_offset = i * red_dim_units0;

        emplace_core_args(reader_desc0, core, i, src_offset, red_dim_offset, read_size, units);
        if (secondary_desc0.has_value()) {
            emplace_core_args(*secondary_desc0, core, i, src_offset, red_dim_offset, read_size, units);
        }
    }

    desc.kernels.push_back(std::move(reader_desc0));
    if (secondary_desc0.has_value()) {
        desc.kernels.push_back(std::move(*secondary_desc0));
    }

    if (num_cores1 > 0) {
        KernelDescriptor reader_desc1 = make_primary(cores1);
        std::optional<KernelDescriptor> secondary_desc1;
        if (use_secondary_dm) {
            secondary_desc1 = make_secondary(cores1, src_cb_page_size1);
        }

        const uint32_t src_offset1 = static_cast<uint32_t>(src_read_size0 * num_cores0);
        const uint32_t red_dim_offset1 = static_cast<uint32_t>(red_dim_units0 * num_cores0);

        for (uint32_t i = 0; i < num_cores1; ++i) {
            const CoreCoord& core = cores_coords1.at(i);
            const bool is_last = (i == num_cores1 - 1);
            const uint32_t core_index = static_cast<uint32_t>(num_cores0 + i);
            const uint32_t read_size = static_cast<uint32_t>(is_last ? src_read_size_last1 : src_read_size1);
            const uint32_t units = is_last ? red_dim_units_last1 : red_dim_units1;
            const uint32_t src_offset = static_cast<uint32_t>(src_offset1 + (i * src_read_size1));
            const uint32_t red_dim_offset = red_dim_offset1 + (i * red_dim_units1);

            emplace_core_args(reader_desc1, core, core_index, src_offset, red_dim_offset, read_size, units);
            if (secondary_desc1.has_value()) {
                emplace_core_args(*secondary_desc1, core, core_index, src_offset, red_dim_offset, read_size, units);
            }
        }

        desc.kernels.push_back(std::move(reader_desc1));
        if (secondary_desc1.has_value()) {
            desc.kernels.push_back(std::move(*secondary_desc1));
        }
    }

    return desc;
}

}  // namespace ttnn::prim
