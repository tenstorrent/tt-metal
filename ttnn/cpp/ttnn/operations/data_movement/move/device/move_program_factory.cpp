// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/work_split.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/move/device/move_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include <algorithm>

using namespace tt::constants;

namespace ttnn::operations::data_movement {

std::vector<CoreRange> get_multicast_regions(const Device* device,
                                             const CoreRangeSet& all_cores,
                                             const CoreCoord& logical_controller) {
    TT_ASSERT(0 < all_cores.ranges().size() and all_cores.ranges().size() <= 2);
    CoreCoord logical_zero = {0, 0};
    TT_ASSERT(logical_controller == logical_zero);

    std::vector<CoreRange> logical_core_ranges;
    auto split_core_range_containing_controller = [&](const CoreRange& controller_core_range) {
        TT_ASSERT(controller_core_range.start_coord == logical_controller);
        CoreRange right_block(CoreCoord(logical_controller.x + 1, logical_controller.y),
                              controller_core_range.end_coord);
        CoreRange remaining_stick = CoreRange(CoreCoord(logical_controller.x, logical_controller.y + 1),
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

// This variant of move is invoked when the input buffer and output buffer overlap, which is possible because input
// buffer is deallocated before the op runs. In this case, data in each core needs to be moved to a temporary local
// location before being copied into the output buffer
operation::ProgramWithCallbacks move_multi_core_with_overlap(const Tensor& input, Tensor& output) {
    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.get_dtype());

    bool tilized = input.get_layout() == Layout::TILE;

    uint32_t page_size = input.buffer()->page_size();

    uint32_t num_pages = tilized ? output.volume() / TILE_HW : output.volume() / output.get_legacy_shape()[-1];
    tt::tt_metal::Device* device = output.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    const auto num_dram_banks = device->num_banks(BufferType::DRAM);
    const auto num_l1_banks = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    uint32_t size_per_l1_bank = tt::tt_metal::detail::SizeBytesPerBank(
        output.buffer()->size(), output.buffer()->page_size(), num_l1_banks, hal.get_alignment(HalMemType::L1));

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
    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

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
        CoreRange noc_cr(device->worker_core_from_logical_core(logical_cr.start_coord),
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
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        bool is_controller = (i == 0);
        std::vector<uint32_t> runtime_args = {src_buffer->address(),
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
                                              const Program& program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

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

// Sharded buffers are mapped to CBs. Move from top of src CB to dst CB
operation::ProgramWithCallbacks move_multi_core_sharded(const Tensor& input, Tensor& output) {
    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.get_dtype());
    auto shard_spec = input.shard_spec().value();
    auto shard_shape = shard_spec.shape;
    auto shard_grid = shard_spec.grid;
    auto input_shape = input.get_legacy_shape();
    auto input_dtype = input.get_dtype();
    auto input_layout = input.get_layout();
    TT_FATAL(input_layout == output.get_layout() && input_dtype == output.get_dtype() &&
                 shard_shape == output.shard_spec().value().shape && input_shape == output.get_legacy_shape(),
             "Error");
    const uint32_t src_cb_sharded = tt::CB::c_in0;
    const uint32_t dst_cb_sharded = tt::CB::c_in1;
    uint32_t tile_size_bytes = tile_size(cb_data_format);
    uint32_t shard_shape_num_tiles = tt::div_up(shard_shape[0] * shard_shape[1], TILE_HEIGHT * TILE_WIDTH);
    uint32_t total_size_bytes = 0;
    uint32_t page_size_bytes = 0;
    if ((shard_shape[0] * shard_shape[1]) % (TILE_HEIGHT * TILE_WIDTH) == 0) {
        uint32_t tile_size_bytes = tile_size(cb_data_format);
        total_size_bytes = shard_shape_num_tiles * tile_size_bytes;
        page_size_bytes = tile_size_bytes;
    } else {
        uint32_t datum_size_bytes = datum_size(cb_data_format);
        total_size_bytes = shard_shape[0] * shard_shape[1] * datum_size_bytes;
        page_size_bytes = shard_shape[1] * datum_size_bytes;
    }
    CircularBufferConfig src_cb_sharded_config =
        CircularBufferConfig(total_size_bytes, {{src_cb_sharded, cb_data_format}})
            .set_page_size(src_cb_sharded, page_size_bytes);
    src_cb_sharded_config.set_globally_allocated_address(*input.buffer());
    auto src_sharded_cb = tt::tt_metal::CreateCircularBuffer(program, shard_grid, src_cb_sharded_config);

    CircularBufferConfig dst_cb_sharded_config =
        CircularBufferConfig(total_size_bytes, {{dst_cb_sharded, cb_data_format}})
            .set_page_size(dst_cb_sharded, page_size_bytes);
    dst_cb_sharded_config.set_globally_allocated_address(*output.buffer());
    auto dst_sharded_cb = tt::tt_metal::CreateCircularBuffer(program, shard_grid, dst_cb_sharded_config);

    auto input_buffer_address = input.buffer()->address();
    auto output_buffer_address = output.buffer()->address();
    TT_FATAL(output_buffer_address > input_buffer_address,
             "Expected output buffer to be allocated at a higher address than input buffer");
    uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
    TT_FATAL(input.buffer()->alignment() == output.buffer()->alignment(),
             "Expected input buffer alignment ({} B) and output buffer alignment ({} B) to be equal",
             input.buffer()->alignment(),
             output.buffer()->alignment());
    TT_FATAL(move_chunk_size_bytes % input.buffer()->alignment() == 0,
             "Expected chunk size bytes to move to be {} byte aligned.",
             input.buffer()->alignment());
    uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
    uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;

    std::vector<uint32_t> reader_compile_time_args = {src_cb_sharded, dst_cb_sharded};
    KernelHandle kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp",
        shard_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = reader_compile_time_args});
    std::vector<uint32_t> runtime_args = {
        total_size_bytes, num_chunks, move_chunk_size_bytes, remainder_chunk_size_bytes};
    SetRuntimeArgs(program, kernel_id, shard_grid, runtime_args);

    const auto& cores = corerange_to_cores(shard_grid, std::nullopt, true);
    auto override_runtime_args_callback = [shard_grid = shard_grid,
                                           kernel_id = kernel_id,
                                           src_sharded_cb = src_sharded_cb,
                                           dst_sharded_cb = dst_sharded_cb,
                                           total_size_bytes = total_size_bytes,
                                           cores = cores](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        UpdateDynamicCircularBufferAddress(program, src_sharded_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, dst_sharded_cb, *dst_buffer);
        auto input_buffer_address = src_buffer->address();
        auto output_buffer_address = dst_buffer->address();
        uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
        uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
        uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;
        std::vector<uint32_t> new_runtime_args = {
            total_size_bytes, num_chunks, move_chunk_size_bytes, remainder_chunk_size_bytes};
        auto& runtime_args_by_core = GetRuntimeArgs(program, kernel_id);
        for (const auto& core : cores) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            std::copy(new_runtime_args.begin(), new_runtime_args.end(), runtime_args.data());
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks move_multi_core(const Tensor& input, Tensor& output) {
    bool src_and_dst_in_l1 = input.memory_config().is_l1() && output.memory_config().is_l1();
    return copy_multi_core(input, output, src_and_dst_in_l1);
}

}  // namespace ttnn::operations::data_movement
