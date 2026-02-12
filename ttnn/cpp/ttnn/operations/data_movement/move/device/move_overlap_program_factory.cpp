// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation_types.hpp"
#include "move_overlap_program_factory.hpp"

#include <cmath>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::prim {

namespace {

std::vector<CoreRange> get_multicast_regions(const CoreRangeSet& all_cores, const CoreCoord& logical_controller) {
    TT_ASSERT(!all_cores.ranges().empty() and all_cores.ranges().size() <= 2);
    const CoreCoord logical_zero = {0, 0};
    TT_ASSERT(logical_controller == logical_zero);

    std::vector<CoreRange> logical_core_ranges;
    auto split_core_range_containing_controller = [&](const CoreRange& controller_core_range) {
        TT_ASSERT(controller_core_range.start_coord == logical_controller);
        CoreRange right_block(
            CoreCoord(logical_controller.x + 1, logical_controller.y), controller_core_range.end_coord);
        CoreRange remaining_stick = CoreRange(
            CoreCoord(logical_controller.x, logical_controller.y + 1),
            CoreCoord(logical_controller.x, controller_core_range.end_coord.y));

        logical_core_ranges.push_back(right_block);
        logical_core_ranges.push_back(remaining_stick);
    };

    CoreRange range_0 = *all_cores.ranges().begin();
    if (all_cores.ranges().size() == 1) {
        split_core_range_containing_controller(range_0);
    } else {
        CoreRange range_1 = *all_cores.ranges().rbegin();
        if (range_0.start_coord == logical_controller) {
            split_core_range_containing_controller(range_0);
            logical_core_ranges.push_back(range_1);
        } else if (range_1.start_coord == logical_controller) {
            split_core_range_containing_controller(range_1);
            logical_core_ranges.push_back(range_0);
        } else {
            TT_THROW("Core {} is not included in set of core ranges!", logical_controller.str());
        }
    }

    TT_ASSERT(logical_core_ranges.size() == 2 or logical_core_ranges.size() == 3);
    return logical_core_ranges;
}

}  // namespace

MoveOverlapProgramFactory::cached_program_t MoveOverlapProgramFactory::create(
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::constants;

    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const bool tilized = input.layout() == Layout::TILE;
    const uint32_t page_size = input.buffer()->page_size();

    const uint32_t num_pages =
        tilized ? (output.physical_volume() / TILE_HW) : (output.physical_volume() / output.padded_shape()[-1]);
    const tt::tt_metal::IDevice* device = output.device();
    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    const auto num_l1_banks = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    const uint32_t size_per_l1_bank = tt::tt_metal::detail::calculate_bank_size_spread(
        output.buffer()->size(), output.buffer()->page_size(), num_l1_banks, tt::tt_metal::hal::get_l1_alignment());

    // CB is being used as temp L1 buffer to copy src data into before writing to dst
    uint32_t cb_index = 0;
    const uint32_t aligned_page_size = round_up_to_mul32(page_size);
    const tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(size_per_l1_bank, {{cb_index, cb_data_format}})
            .set_page_size(cb_index, aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    auto semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    std::vector<uint32_t> compile_time_args = {cb_index};
    if (!tilized) {
        compile_time_args.push_back(page_size);
    }
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    const std::string kernel_path =
        tilized
            ? "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/"
              "move_stick_layout_interleaved_with_overlap.cpp";

    const tt::tt_metal::KernelHandle kernel_id = tt::tt_metal::CreateKernel(
        program, kernel_path, all_cores, tt::tt_metal::DataMovementConfig{.compile_args = compile_time_args});

    const CoreCoord logical_controller = CoreCoord{0, 0};
    const CoreCoord noc_controller = device->worker_core_from_logical_core(logical_controller);
    std::vector<CoreRange> logical_multicast_regions = get_multicast_regions(all_cores, logical_controller);

    std::vector<CoreRange> noc_multicast_regions;
    for (const auto& logical_cr : logical_multicast_regions) {
        const CoreRange noc_cr(
            device->worker_core_from_logical_core(logical_cr.start_coord),
            device->worker_core_from_logical_core(logical_cr.end_coord));
        noc_multicast_regions.push_back(noc_cr);
    }

    const CoreRange range_0_noc = noc_multicast_regions[0];
    const CoreRange range_1_noc = noc_multicast_regions[1];
    const bool do_third_multicast = (noc_multicast_regions.size() == 3);

    for (uint32_t i = 0, pages_handled_per_core = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        const bool is_controller = (i == 0);
        std::vector<uint32_t> runtime_args = {
            src_buffer->address(),
            dst_buffer->address(),
            pages_handled_per_core,
            num_pages_per_core,
            semaphore_id,
            (uint32_t)noc_controller.x,
            (uint32_t)noc_controller.y,
            /*control_value=*/(num_cores - 1),
            (uint32_t)is_controller,
            (uint32_t)range_0_noc.start_coord.x,
            (uint32_t)range_0_noc.start_coord.y,
            (uint32_t)range_0_noc.end_coord.x,
            (uint32_t)range_0_noc.end_coord.y,
            (uint32_t)logical_multicast_regions[0].size(),
            (uint32_t)range_1_noc.start_coord.x,
            (uint32_t)range_1_noc.start_coord.y,
            (uint32_t)range_1_noc.end_coord.x,
            (uint32_t)range_1_noc.end_coord.y,
            (uint32_t)logical_multicast_regions[1].size(),
            (uint32_t)noc_multicast_regions.back().start_coord.x,
            (uint32_t)noc_multicast_regions.back().start_coord.y,
            (uint32_t)noc_multicast_regions.back().end_coord.x,
            (uint32_t)noc_multicast_regions.back().end_coord.y,
            (uint32_t)logical_multicast_regions.back().size(),
            (uint32_t)do_third_multicast};
        if (!tilized) {
            runtime_args.push_back(aligned_page_size);
        }
        SetRuntimeArgs(program, kernel_id, core, runtime_args);
        pages_handled_per_core += num_pages_per_core;
    }

    return {
        std::move(program),
        MoveOverlapProgramFactory::shared_variables_t{.reader_kernel_id = kernel_id, .num_cores = num_cores}};
}

void MoveOverlapProgramFactory::override_runtime_arguments(
    MoveOverlapProgramFactory::cached_program_t& cached_program,
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const Tensor& input = tensor_args.input_tensor;
    Tensor& output = tensor_return_value;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const tt::tt_metal::KernelHandle reader_kernel_id = cached_program.shared_variables.reader_kernel_id;

    const CoreCoord compute_with_storage_grid_size = output.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (const CoreCoord& core : cores) {
        auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        runtime_args[1] = dst_buffer->address();
    }
}

}  // namespace ttnn::prim
