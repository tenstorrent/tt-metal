// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/move/device/move_multi_core_with_overlap_program_factory.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/allocator.hpp>
#include <algorithm>

#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

extern std::vector<CoreRange> get_multicast_regions(
    const IDevice* device, const CoreRangeSet& all_cores, const CoreCoord& logical_controller);

// This variant of move is invoked when the input buffer and output buffer overlap, which is possible because input
// buffer is deallocated before the op runs. In this case, data in each core needs to be moved to a temporary local
// location before being copied into the output buffer
operation::ProgramWithCallbacks move_multi_core_with_overlap(const Tensor& input, Tensor& output) {
    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());

    bool tilized = input.layout() == Layout::TILE;

    uint32_t page_size = input.buffer()->page_size();

    uint32_t num_pages =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.padded_shape()[-1];
    tt::tt_metal::IDevice* device = output.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    const auto num_dram_banks = device->allocator()->get_num_banks(BufferType::DRAM);
    const auto num_l1_banks = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    uint32_t size_per_l1_bank = tt::tt_metal::detail::SizeBytesPerBank(
        output.buffer()->size(), output.buffer()->page_size(), num_l1_banks, hal::get_l1_alignment());

    // CB is being used as temp L1 buffer to copy src data into before writing to dst
    uint32_t cb_index = 0;
    uint32_t aligned_page_size = round_up_to_mul32(page_size);
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(size_per_l1_bank, {{cb_index, cb_data_format}})
            .set_page_size(cb_index, aligned_page_size);
    auto cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    auto semaphore_id = CreateSemaphore(program, all_cores, 0);

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;

    uint32_t log2_page_size = 0;
    std::vector<uint32_t> compile_time_args = {cb_index, (uint32_t)src_is_dram, (uint32_t)dst_is_dram};
    if (!tilized) {
        bool page_size_is_power_of_two = is_power_of_two_at_least_32(page_size);
        log2_page_size = page_size_is_power_of_two ? (std::uint32_t)log2(page_size) : 0;
        compile_time_args.push_back((uint32_t)page_size_is_power_of_two);
    }

    KernelHandle kernel_id = CreateKernel(
        program,
        tilized
            ? "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/"
              "move_stick_layout_interleaved_with_overlap.cpp",
        all_cores,
        DataMovementConfig{.compile_args = compile_time_args});

    const CoreCoord logical_controller = CoreCoord{0, 0};
    CoreCoord noc_controller = device->worker_core_from_logical_core(logical_controller);
    std::vector<CoreRange> logical_multicast_regions = get_multicast_regions(device, all_cores, logical_controller);

    std::vector<CoreRange> noc_multicast_regions;
    for (const auto& logical_cr : logical_multicast_regions) {
        CoreRange noc_cr(
            device->worker_core_from_logical_core(logical_cr.start_coord),
            device->worker_core_from_logical_core(logical_cr.end_coord));
        noc_multicast_regions.push_back(std::move(noc_cr));
    }

    CoreRange range_0_noc = noc_multicast_regions[0];
    CoreRange range_1_noc = noc_multicast_regions[1];
    // if third multicast is not needed range_2_noc will be ignored
    bool do_third_multicast = (noc_multicast_regions.size() == 3);

    uint32_t total_num_pages = 0;
    for (uint32_t i = 0, pages_handled_per_core = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        bool is_controller = (i == 0);
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
            runtime_args.push_back(page_size);
            runtime_args.push_back(aligned_page_size);
            runtime_args.push_back(log2_page_size);
        }
        SetRuntimeArgs(program, kernel_id, core, runtime_args);
        pages_handled_per_core += num_pages_per_core;
    }

    auto override_runtime_args_callback = [kernel_id, num_cores, num_cores_y](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement
