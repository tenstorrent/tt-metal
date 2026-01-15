// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "grid_sample_utils.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/pool/grid_sample/device/grid_sample_bilinear_program_factory.hpp"

namespace ttnn::operations::pool::grid_sample::program {

GridSampleBilinearProgramFactory::cached_program_t GridSampleBilinearProgramFactory::create(
    const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& grid_tensor = tensor_args.grid;
    bool use_precomputed_grid = operation_attributes.use_precomputed_grid;
    bool batch_output_channels = operation_attributes.batch_output_channels;

    const bool is_sharded = grid_tensor.is_sharded();
    tt::tt_metal::Program program{};

    // Data formats and device
    const auto [input_cb_data_format, grid_cb_data_format, output_cb_data_format] = std::make_tuple(
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype()));
    tt::tt_metal::IDevice* const device = output_tensor.device();

    // Shape and dimensions
    const auto& [input_shape, grid_shape, output_shape] =
        std::tie(input_tensor.padded_shape(), grid_tensor.padded_shape(), output_tensor.padded_shape());
    const uint32_t input_batch = input_shape[0], input_height = input_shape[1], input_width = input_shape[2];
    const uint32_t grid_height = grid_shape[1], grid_width = grid_shape[2];
    const uint32_t grid_hw = grid_height * grid_width;
    const uint32_t grid_batching_factor = get_grid_batching_factor(grid_tensor, use_precomputed_grid);
    const bool enable_split_reader =
        should_use_split_reader(input_tensor, grid_tensor, use_precomputed_grid, "bilinear");

    tt::tt_metal::CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores, grid_nsticks_per_core, output_nsticks_per_core = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;
    // Calculate total work and determine actual cores needed
    const uint32_t total_grid_nsticks = grid_tensor.physical_volume() / grid_shape[-1];

    if (is_sharded) {
        const auto grid_shard_spec = grid_tensor.shard_spec().value();
        grid_nsticks_per_core = grid_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];

        num_cores = (total_grid_nsticks + grid_nsticks_per_core - 1) / grid_nsticks_per_core;
        all_cores = grid_shard_spec.grid;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, grid_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_grid_nsticks);

        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        grid_nsticks_per_core = num_sticks_1;

        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    uint32_t cb_idx = tt::CBIndex::c_0;

    // Create CBs
    const uint32_t grid_stick_size =
        is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);
    const auto [grid_cb_index, grid_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++,
        program,
        all_cores,
        grid_stick_size,
        is_sharded ? grid_nsticks_per_core : 1,
        grid_cb_data_format,
        is_sharded ? grid_tensor.buffer() : nullptr);

    const uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[-1] / tt::constants::TILE_WIDTH);
    const uint32_t input_cb_page_size = in_ntiles_c * tt::constants::TILE_HW * input_tensor.element_size();
    const auto [input_cb_index_0, input_cb_handle_0] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, input_cb_page_size, BUFFERING_FACTOR, input_cb_data_format);

    uint32_t input_cb_index_1 = DUMMY_CB_ID;
    tt::tt_metal::CBHandle input_cb_handle_1 = 0;
    if (is_sharded && enable_split_reader) {
        std::tie(input_cb_index_1, input_cb_handle_1) = tt::tt_metal::create_cb(
            cb_idx++, program, all_cores, input_cb_page_size, BUFFERING_FACTOR, input_cb_data_format);
    }

    const uint32_t scalar_cb_page_size = tt::tile_size(input_cb_data_format);
    const auto [scalar_cb_index_0, scalar_cb_handle_0] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, scalar_cb_page_size, BUFFERING_FACTOR, input_cb_data_format);

    uint32_t scalar_cb_index_1 = DUMMY_CB_ID;
    tt::tt_metal::CBHandle scalar_cb_handle_1 = 0;
    if (is_sharded && enable_split_reader) {
        std::tie(scalar_cb_index_1, scalar_cb_handle_1) = tt::tt_metal::create_cb(
            cb_idx++, program, all_cores, scalar_cb_page_size, BUFFERING_FACTOR, input_cb_data_format);
    }

    const uint32_t out_ntiles_c = (uint32_t)std::ceil((float)output_shape[-1] / tt::constants::FACE_WIDTH);
    const uint32_t output_cb_page_size = tt::constants::FACE_WIDTH * output_tensor.element_size();
    const uint32_t output_cb_pages =
        is_sharded ? output_nsticks_per_core * out_ntiles_c : out_ntiles_c * BUFFERING_FACTOR;
    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++,
        program,
        all_cores,
        output_cb_page_size,
        output_cb_pages,
        output_cb_data_format,
        is_sharded ? output_tensor.buffer() : nullptr);

    // Prepare stick size arguments with proper names
    const uint32_t input_stick_size = get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t grid_stick_size_arg =
        is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);

    // Reader compile-time arguments - shared arguments first, then specific ones
    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index_0,                            // ct_arg[0]: input_cb_index_0
        grid_cb_index,                               // ct_arg[1]: grid_cb_index
        scalar_cb_index_0,                           // ct_arg[2]: scalar_cb_index_0
        input_stick_size,                            // ct_arg[3]: input_stick_size
        grid_stick_size_arg,                         // ct_arg[4]: grid_stick_size
        input_batch,                                 // ct_arg[5]: input_batch
        input_height,                                // ct_arg[6]: input_height
        input_width,                                 // ct_arg[7]: input_width
        grid_batching_factor,                        // ct_arg[8]: grid_batching_factor (shared)
        static_cast<uint32_t>(grid_tensor.dtype()),  // ct_arg[9]: grid_dtype (shared)
        grid_hw,                                     // ct_arg[10]: grid_hw (shared)
        use_precomputed_grid ? 1U : 0U               // ct_arg[11]: use_precomputed_grid (shared)
    };

    if (is_sharded) {
        reader_compile_time_args.push_back(enable_split_reader ? 1U : 0U);  // ct_arg[12]: enable_split_reader
        reader_compile_time_args.push_back(0U);  // ct_arg[13]: reader_id (will be set later per reader)
        reader_compile_time_args.push_back(grid_nsticks_per_core);  // ct_arg[14]: grid_nsticks_per_core
    }

    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);
    if (!is_sharded) {
        tt::tt_metal::TensorAccessorArgs(*grid_tensor.buffer()).append_to(reader_compile_time_args);
    }

    const std::string reader_kernel_path =
        is_sharded ? "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_sharded.cpp"
                   : "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/"
                     "reader_grid_sample_interleaved_start_id.cpp";

    // Create kernels
    auto create_reader_config = [&](const std::vector<uint32_t>& args, auto processor, auto noc) {
        return tt::tt_metal::DataMovementConfig{.processor = processor, .noc = noc, .compile_args = args};
    };

    tt::tt_metal::KernelHandle reader0_kernel_id, reader1_kernel_id = 0;
    if (is_sharded) {
        auto reader0_compile_time_args = reader_compile_time_args;
        reader0_compile_time_args[13] = 0;  // ct_arg[13]: reader_id = 0

        reader0_kernel_id = tt::tt_metal::CreateKernel(
            program,
            reader_kernel_path,
            all_cores,
            create_reader_config(
                reader0_compile_time_args,
                tt::tt_metal::DataMovementProcessor::RISCV_0,
                tt::tt_metal::NOC::RISCV_0_default));

        if (enable_split_reader) {
            auto reader1_compile_time_args = reader_compile_time_args;
            reader1_compile_time_args[0] = input_cb_index_1;   // ct_arg[0]: input_cb_index_1
            reader1_compile_time_args[2] = scalar_cb_index_1;  // ct_arg[2]: scalar_cb_index_1
            reader1_compile_time_args[13] = 1;                 // ct_arg[13]: reader_id = 1

            reader1_kernel_id = tt::tt_metal::CreateKernel(
                program,
                reader_kernel_path,
                all_cores,
                create_reader_config(
                    reader1_compile_time_args,
                    tt::tt_metal::DataMovementProcessor::RISCV_1,
                    tt::tt_metal::NOC::RISCV_1_default));
        }
    } else {
        reader0_kernel_id = tt::tt_metal::CreateKernel(
            program, reader_kernel_path, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    }

    const bool is_output_tiled = false;
    const bool is_output_block_format = false;
    const uint32_t pre_tilize_cb_id =
        32;  // Unused CB for pool compute kernel for grid sample, we don't have tiled output in gridsample

    // Compute kernels
    const uint32_t channels_per_shard = input_shape[-1];
    const uint32_t in_nblocks_c = (uint32_t)std::ceil((float)in_ntiles_c / MAX_TILES_PER_REDUCTION);

    auto create_compute_kernel = [&](tt::tt_metal::CoreRangeSet cores, uint32_t total_interpolations) {
        // Compute kernel compile-time arguments
        std::vector<uint32_t> compute_compile_time_args = {
            in_ntiles_c,                       // ct_arg[0]: in_ntiles_c
            REDUCTION_SIZE,                    // ct_arg[1]: REDUCTION_SIZE
            enable_split_reader,               // ct_arg[2]: enable_split_reader
            total_interpolations,              // ct_arg[3]: total_interpolations
            channels_per_shard,                // ct_arg[4]: channels_per_shard
            in_nblocks_c,                      // ct_arg[5]: in_nblocks_c
            MAX_ROWS_FOR_REDUCTION,            // ct_arg[6]: MAX_ROWS_FOR_REDUCTION
            input_cb_index_0,                  // ct_arg[7]: input_cb_index_0
            input_cb_index_1,                  // ct_arg[8]: input_cb_index_1
            scalar_cb_index_0,                 // ct_arg[9]: scalar_cb_index_0
            scalar_cb_index_1,                 // ct_arg[10]: scalar_cb_index_1
            DUMMY_CB_ID,                       // ct_arg[11]: unused
            DUMMY_CB_ID,                       // ct_arg[12]: unused
            DUMMY_CB_ID,                       // ct_arg[13]: unused
            DUMMY_CB_ID,                       // ct_arg[14]: unused
            DUMMY_CB_ID,                       // ct_arg[15]: unused
            DUMMY_CB_ID,                       // ct_arg[16]: unused
            output_cb_index,                   // ct_arg[17]: output_cb_index
            DUMMY_CB_ID,                       // ct_arg[18]: unused
            ONE_SCALAR_PER_CORE,               // ct_arg[19]: ONE_SCALAR_PER_CORE
            pre_tilize_cb_id,                  // ct_arg[20]: pre_tilize_cb_id
            is_output_tiled ? 1U : 0U,         // ct_arg[21]: is_output_tiled
            is_output_block_format ? 1U : 0U,  // ct_arg[22]: is_output_block_format
            false,                             // ct_arg[23]: return_indices (unused)
            1,                                 // ct_arg[24]: stride_h (unused)
            1,                                 // ct_arg[25]: stride_w (unused)
            1,                                 // ct_arg[26]: in_h_padded (unused)
            1,                                 // ct_arg[27]: in_w_padded (unused)
            1,                                 // ct_arg[28]: eff_kernel_h (unused)
            1,                                 // ct_arg[29]: eff_kernel_w (unused)
            1,                                 // ct_arg[30]: pad_l (unused)
        };

        return tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp",
            cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_compile_time_args,
                .defines = get_defines(pool::Pool2DType::AVG_POOL2D)});
    };

    if (is_sharded || core_group_1.num_cores() > 0) {
        create_compute_kernel(
            is_sharded ? all_cores : core_group_1,
            grid_batching_factor * (is_sharded ? grid_nsticks_per_core : num_sticks_per_core_group_1));
    }
    if (!is_sharded && core_group_2.num_cores() > 0) {
        create_compute_kernel(core_group_2, grid_batching_factor * num_sticks_per_core_group_2);
    }

    // Writer kernel (interleaved only)
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    if (!is_sharded) {
        // Writer compile-time arguments - expanded row by row for readability
        std::vector<uint32_t> writer_compile_time_args = {
            output_cb_index,                                      // ct_arg[0]: output_cb_index
            get_aligned_stick_size(output_shape, output_tensor),  // ct_arg[1]: output_stick_size
            out_ntiles_c                                          // ct_arg[2]: out_ntiles_c
        };
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }

    // Set runtime arguments
    if (is_sharded) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            // Runtime arguments for sharded reader
            std::vector<uint32_t> reader_runtime_args = {
                input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
                i * grid_nsticks_per_core          // rt_arg[1]: grid_stick_offset
            };

            tt::tt_metal::SetRuntimeArgs(program, reader0_kernel_id, core, reader_runtime_args);
            if (enable_split_reader) {
                tt::tt_metal::SetRuntimeArgs(program, reader1_kernel_id, core, reader_runtime_args);
            }
        }
    } else {
        uint32_t grid_processed = 0;
        uint32_t output_processed = 0;

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t grid_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            const uint32_t output_sticks = batch_output_channels ? grid_sticks : grid_sticks * grid_batching_factor;

            // Runtime arguments for interleaved reader - expanded row by row
            std::vector<uint32_t> reader_runtime_args = {
                input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
                grid_tensor.buffer()->address(),   // rt_arg[1]: grid_buffer_address
                grid_sticks,                       // rt_arg[2]: grid_sticks
                grid_processed                     // rt_arg[3]: grid_processed
            };

            // Runtime arguments for interleaved writer - expanded row by row
            std::vector<uint32_t> writer_runtime_args = {
                output_tensor.buffer()->address(),  // rt_arg[0]: output_buffer_address
                output_sticks,                      // rt_arg[1]: output_sticks
                output_processed                    // rt_arg[2]: output_processed
            };

            tt::tt_metal::SetRuntimeArgs(program, reader0_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

            grid_processed += grid_sticks;
            output_processed += output_sticks;
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .is_sharded = is_sharded,
            .logical_cores = logical_cores,
            .grid_cb_handle = grid_cb_handle,
            .output_cb_handle = output_cb_handle,
            .num_cores = num_cores,
            .enable_split_reader = enable_split_reader,
            .reader0_kernel_id = reader0_kernel_id,
            .reader1_kernel_id = reader1_kernel_id,
            .writer_kernel_id = writer_kernel_id,
        }};
}

void GridSampleBilinearProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const GridSampleParams& /*operation_attributes*/,
    const GridSampleInputs& tensor_args,
    Tensor& output_tensor) {
    auto& prog = cached_program.program;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& grid_tensor = tensor_args.grid;
    const auto& is_sharded = cached_program.shared_variables.is_sharded;
    const auto& grid_cb_handle = cached_program.shared_variables.grid_cb_handle;
    const auto& output_cb_handle = cached_program.shared_variables.output_cb_handle;
    const auto& num_cores = cached_program.shared_variables.num_cores;
    const auto& logical_cores = cached_program.shared_variables.logical_cores;
    const auto& reader0_kernel_id = cached_program.shared_variables.reader0_kernel_id;
    const auto& reader1_kernel_id = cached_program.shared_variables.reader1_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& enable_split_reader = cached_program.shared_variables.enable_split_reader;

    if (is_sharded) {
        tt::tt_metal::UpdateDynamicCircularBufferAddress(prog, grid_cb_handle, *grid_tensor.buffer());
        tt::tt_metal::UpdateDynamicCircularBufferAddress(prog, output_cb_handle, *output_tensor.buffer());

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            tt::tt_metal::GetRuntimeArgs(prog, reader0_kernel_id, core)[0] = input_tensor.buffer()->address();
            if (enable_split_reader) {
                tt::tt_metal::GetRuntimeArgs(prog, reader1_kernel_id, core)[0] = input_tensor.buffer()->address();
            }
        }
    } else {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(prog, reader0_kernel_id, core);
            reader_runtime_args[0] = input_tensor.buffer()->address();  // rt_arg[0]: input_buffer_address
            reader_runtime_args[1] = grid_tensor.buffer()->address();   // rt_arg[1]: grid_buffer_address
            tt::tt_metal::GetRuntimeArgs(prog, writer_kernel_id, core)[0] =
                output_tensor.buffer()->address();  // rt_arg[0]: output_buffer_address
        }
    }
}

}  // namespace ttnn::operations::pool::grid_sample::program
