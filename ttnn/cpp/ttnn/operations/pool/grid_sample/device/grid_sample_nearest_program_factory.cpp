// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "grid_sample_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor GridSampleNearestProgramFactory::create_descriptor(
    const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& grid_tensor = tensor_args.grid;
    const bool use_precomputed_grid = operation_attributes.use_precomputed_grid;
    const bool align_corners = operation_attributes.align_corners;

    const bool is_sharded = grid_tensor.is_sharded();
    ProgramDescriptor desc;

    // Data formats and device
    const auto [input_cb_data_format, grid_cb_data_format, output_cb_data_format] = std::make_tuple(
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype()));
    tt::tt_metal::IDevice* const device = output_tensor.device();

    // Shape and dimensions
    const auto& [input_shape, grid_shape, output_shape] =
        std::tie(input_tensor.padded_shape(), grid_tensor.padded_shape(), output_tensor.padded_shape());
    const uint32_t input_height = input_shape[1], input_width = input_shape[2];
    const uint32_t grid_height = grid_shape[1], grid_width = grid_shape[2];
    const uint32_t grid_hw = grid_height * grid_width;
    const uint32_t grid_batching_factor = get_grid_batching_factor(grid_tensor, use_precomputed_grid, "nearest");
    const bool enable_split_reader =
        should_use_split_reader(input_tensor, grid_tensor, use_precomputed_grid, "nearest");
    tt::tt_metal::CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores, grid_nsticks_per_core, output_nsticks_per_core = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;

    if (is_sharded) {
        const auto grid_shard_spec = grid_tensor.shard_spec().value();
        all_cores = grid_shard_spec.grid;
        num_cores = grid_shard_spec.num_cores();
        grid_nsticks_per_core = grid_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        logical_cores = corerange_to_cores(
            all_cores, num_cores, grid_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        uint32_t grid_nsticks = grid_tensor.physical_volume() / grid_shape[-1];
        if (output_tensor.shard_spec().has_value()) {
            grid_nsticks = output_tensor.shard_spec().value().shape[0] * output_tensor.shard_spec().value().num_cores();
        } else {
            grid_nsticks = tt::round_up(grid_nsticks, compute_grid_size.x * compute_grid_size.y);
        }
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, grid_nsticks);

        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        grid_nsticks_per_core = num_sticks_1;
        output_nsticks_per_core = num_sticks_1;
        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    uint32_t cb_idx = tt::CBIndex::c_0;

    // Create CBs
    const uint32_t grid_stick_size =
        is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);
    const uint32_t grid_cb_index0 = cb_idx++;
    const uint32_t grid_cb_num_pages0 = is_sharded ? grid_nsticks_per_core : 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = grid_cb_num_pages0 * grid_stick_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(grid_cb_index0),
            .data_format = grid_cb_data_format,
            .page_size = grid_stick_size,
        }}},
        .buffer = is_sharded ? grid_tensor.buffer() : nullptr,
    });

    uint32_t grid_cb_index1 = DUMMY_CB_ID;
    // Nearest mode has support for DRAM interleaved grid and sharded output
    if (enable_split_reader && !is_sharded) {
        grid_cb_index1 = cb_idx++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = grid_stick_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(grid_cb_index1),
                .data_format = grid_cb_data_format,
                .page_size = grid_stick_size,
            }}},
        });
    }

    const uint32_t output_cb_page_size =
        static_cast<uint32_t>(static_cast<float>(output_shape[-1]) * output_tensor.element_size());
    const uint32_t output_cb_pages = output_nsticks_per_core;

    // Fill CB - single page to hold pre-filled stick
    const uint32_t fill_cb_index = cb_idx++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(fill_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_cb_page_size,
        }}},
    });

    const uint32_t output_cb_index = cb_idx++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_pages * output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_cb_page_size,
        }}},
        .buffer = output_tensor.buffer(),
    });

    // Prepare stick size arguments with proper names
    const uint32_t input_stick_size = get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t grid_stick_size_arg =
        is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);

    // Writer for nearest mode with sharded grid - matches sharded reader argument structure
    std::vector<uint32_t> writer_compile_time_args = {
        grid_cb_index0,                              // ct_arg[0]: grid_cb_index
        output_cb_index,                             // ct_arg[1]: output_cb_index (replaces scalar_cb_index)
        input_stick_size,                            // ct_arg[2]: input_stick_size
        grid_stick_size_arg,                         // ct_arg[3]: grid_stick_size
        input_height,                                // ct_arg[4]: input_height
        input_width,                                 // ct_arg[5]: input_width
        grid_batching_factor,                        // ct_arg[6]: grid_batching_factor
        static_cast<uint32_t>(grid_tensor.dtype()),  // ct_arg[7]: grid_dtype
        grid_hw,                                     // ct_arg[8]: grid_hw
        use_precomputed_grid ? 1U : 0U,              // ct_arg[9]: use_precomputed_grid
        align_corners ? 1U : 0U,                     // ct_arg[10]: align_corners
        enable_split_reader ? 1U : 0U,               // ct_arg[11]: split_reader (same as reader)
        0U,                                          // ct_arg[12]: reader_id (will be set per core)
        grid_nsticks_per_core,                       // ct_arg[13]: grid_nsticks_per_core
        is_sharded ? 1U : 0U,                        // ct_arg[14]: is_sharded
        fill_cb_index,                               // ct_arg[15]: fill_cb_index
        input_shape[0]                               // ct_arg[16]: batch_size
    };

    // Add tensor accessor args for input tensor (17 compile time args offset)
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(writer_compile_time_args);
    if (!is_sharded) {
        tt::tt_metal::TensorAccessorArgs(*grid_tensor.buffer()).append_to(writer_compile_time_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(writer_compile_time_args);
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
    };

    KernelDescriptor writer1_desc;
    if (enable_split_reader) {
        auto writer1_compile_time_args = writer_compile_time_args;
        writer1_compile_time_args[0] = is_sharded ? grid_cb_index0 : grid_cb_index1;  // ct_arg[0]: grid_cb_index1
        writer1_compile_time_args[12] = 1;                                            // ct_arg[12]: reader_id = 1

        writer1_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp";
        writer1_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer1_desc.core_ranges = all_cores;
        writer1_desc.compile_time_args = std::move(writer1_compile_time_args);
        writer1_desc.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
        };
    }

    // Set runtime arguments
    if (is_sharded) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            // Runtime arguments for sharded reader
            KernelDescriptor::CoreRuntimeArgs runtime_args = {
                input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
                i * grid_nsticks_per_core          // rt_arg[1]: grid_stick_offset
            };

            writer_desc.runtime_args.emplace_back(core, runtime_args);
            if (enable_split_reader) {
                writer1_desc.runtime_args.emplace_back(core, runtime_args);
            }
        }
    } else {
        uint32_t grid_processed = 0;

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t grid_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

            // Runtime arguments for interleaved reader - expanded row by row
            KernelDescriptor::CoreRuntimeArgs runtime_args = {
                input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
                grid_tensor.buffer()->address(),   // rt_arg[1]: grid_buffer_address
                grid_sticks,                       // rt_arg[2]: grid_sticks
                grid_processed                     // rt_arg[3]: grid_processed
            };

            writer_desc.runtime_args.emplace_back(core, runtime_args);
            if (enable_split_reader) {
                // For split reader in nearest mode, second writer needs same runtime args
                // The kernel logic handles the split internally via reader_id
                writer1_desc.runtime_args.emplace_back(core, runtime_args);
            }

            grid_processed += grid_sticks;
        }
    }

    desc.kernels.push_back(std::move(writer_desc));
    if (enable_split_reader) {
        desc.kernels.push_back(std::move(writer1_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
