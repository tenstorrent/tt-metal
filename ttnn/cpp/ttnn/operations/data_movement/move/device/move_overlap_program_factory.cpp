// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation_types.hpp"
#include "move_overlap_program_factory.hpp"

#include <tt-metalium/experimental/program_descriptor_patching.hpp>

#include <cmath>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

std::vector<CoreRange> get_multicast_regions(const CoreRangeSet& all_cores, const CoreCoord& logical_controller) {
    TT_ASSERT(!all_cores.ranges().empty() and all_cores.ranges().size() <= 2);
    const CoreCoord logical_zero = {0, 0};
    TT_ASSERT(logical_controller == logical_zero);

    std::vector<CoreRange> logical_core_ranges;
    logical_core_ranges.reserve(3);
    // Either split can be empty: the controller's range may be a single row, column, or just itself.
    auto split_core_range_containing_controller = [&](const CoreRange& controller_core_range) {
        TT_ASSERT(controller_core_range.start_coord == logical_controller);
        if (controller_core_range.end_coord.x > logical_controller.x) {
            logical_core_ranges.push_back(
                CoreRange(CoreCoord(logical_controller.x + 1, logical_controller.y), controller_core_range.end_coord));
        }
        if (controller_core_range.end_coord.y > logical_controller.y) {
            logical_core_ranges.push_back(CoreRange(
                CoreCoord(logical_controller.x, logical_controller.y + 1),
                CoreCoord(logical_controller.x, controller_core_range.end_coord.y)));
        }
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

    TT_ASSERT(logical_core_ranges.size() <= 3);
    return logical_core_ranges;
}

}  // namespace

ProgramDescriptor MoveOverlapProgramFactory::create_descriptor(
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::constants;

    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

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

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    ProgramDescriptor desc;

    // CB is being used as temp L1 buffer to copy src data into before writing to dst
    const uint32_t cb_index = 0;
    // Must round the same way as size_per_l1_bank below, or a 16-mod-32 page_size overruns the CB.
    const uint32_t aligned_page_size = tt::align(page_size, tt::tt_metal::hal::get_l1_alignment());
    desc.cbs.push_back(CBDescriptor{
        .total_size = size_per_l1_bank,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = cb_data_format,
            .page_size = aligned_page_size,
        }}},
    });

    // Semaphore used by the controller core to coordinate multicast.
    const uint32_t semaphore_id = 0;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

    std::vector<uint32_t> compile_time_args = {cb_index};
    if (!tilized) {
        compile_time_args.push_back(page_size);
    }
    TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    const std::string kernel_path =
        tilized
            ? "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/"
              "move_stick_layout_interleaved_with_overlap.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(compile_time_args);
    reader_desc.config = DataMovementConfigDescriptor{};

    const CoreCoord logical_controller = CoreCoord{0, 0};
    const CoreCoord noc_controller = device->worker_core_from_logical_core(logical_controller);
    std::vector<CoreRange> logical_multicast_regions = get_multicast_regions(all_cores, logical_controller);

    std::vector<CoreRange> noc_multicast_regions;
    noc_multicast_regions.reserve(logical_multicast_regions.size());
    for (const auto& logical_cr : logical_multicast_regions) {
        const CoreRange noc_cr(
            device->worker_core_from_logical_core(logical_cr.start_coord),
            device->worker_core_from_logical_core(logical_cr.end_coord));
        noc_multicast_regions.push_back(noc_cr);
    }

    // 0-3 regions depending on shard shape; the kernel only multicasts the first num_multicast_regions
    // slots, so unused slots below are just padding.
    const uint32_t num_multicast_regions = static_cast<uint32_t>(noc_multicast_regions.size());
    struct McastRegionArgs {
        uint32_t start_x = 0, start_y = 0, end_x = 0, end_y = 0, size = 0;
    };
    auto region_args = [&](uint32_t index) -> McastRegionArgs {
        if (index >= num_multicast_regions) {
            return {};
        }
        const CoreRange& r = noc_multicast_regions[index];
        return {
            static_cast<uint32_t>(r.start_coord.x),
            static_cast<uint32_t>(r.start_coord.y),
            static_cast<uint32_t>(r.end_coord.x),
            static_cast<uint32_t>(r.end_coord.y),
            static_cast<uint32_t>(logical_multicast_regions[index].size())};
    };
    const McastRegionArgs region_0_args = region_args(0);
    const McastRegionArgs region_1_args = region_args(1);
    const McastRegionArgs region_2_args = region_args(2);

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
        // Buffer* entries register BufferBindings for src/dst addresses (positions 0/1);
        // the framework patches them on cache hits without rebuilding the descriptor.
        KernelDescriptor::RTArgList runtime_args;
        runtime_args.reserve(tilized ? 25 : 26);
        runtime_args.push_back(src_buffer);
        runtime_args.push_back(dst_buffer);
        runtime_args.push_back(pages_handled_per_core);
        runtime_args.push_back(num_pages_per_core);
        runtime_args.push_back(semaphore_id);
        runtime_args.push_back(static_cast<uint32_t>(noc_controller.x));
        runtime_args.push_back(static_cast<uint32_t>(noc_controller.y));
        runtime_args.push_back(num_cores - 1);  // control_value
        runtime_args.push_back(static_cast<uint32_t>(is_controller));
        runtime_args.push_back(region_0_args.start_x);
        runtime_args.push_back(region_0_args.start_y);
        runtime_args.push_back(region_0_args.end_x);
        runtime_args.push_back(region_0_args.end_y);
        runtime_args.push_back(region_0_args.size);
        runtime_args.push_back(region_1_args.start_x);
        runtime_args.push_back(region_1_args.start_y);
        runtime_args.push_back(region_1_args.end_x);
        runtime_args.push_back(region_1_args.end_y);
        runtime_args.push_back(region_1_args.size);
        runtime_args.push_back(region_2_args.start_x);
        runtime_args.push_back(region_2_args.start_y);
        runtime_args.push_back(region_2_args.end_x);
        runtime_args.push_back(region_2_args.end_y);
        runtime_args.push_back(region_2_args.size);
        runtime_args.push_back(num_multicast_regions);
        if (!tilized) {
            runtime_args.push_back(aligned_page_size);
        }
        reader_desc.emplace_runtime_args(core, runtime_args);
        pages_handled_per_core += num_pages_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

void MoveOverlapProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Single reader kernel, src/dst declared at slots 0/1.
    const uint32_t src_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t dst_addr = tensor_return_value.buffer()->address();
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, 0)) {
        for (auto& a : col) {
            if (a.size() < 2) {
                continue;
            }
            a[0] = src_addr;
            a[1] = dst_addr;
        }
    }
}

}  // namespace ttnn::prim
