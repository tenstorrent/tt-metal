// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_program_factory.hpp"

#include "tt-metalium/tt_backend_api_types.hpp"
#include <tt-metalium/hal.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

namespace ttnn::operations::experimental::cnn::detail {

using namespace tt::constants;

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor) {
    const uint32_t element_size_bytes = input_tensor.element_size();
    const uint32_t l1_alignment_bytes = tt::tt_metal::hal::get_l1_alignment();
    return l1_alignment_bytes / element_size_bytes;
}

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_hwc(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // If we are pulling from DRAM we infer the shard shape by using the output shard spec
    const bool is_input_in_dram = a.buffer()->core_type() == tt::CoreType::DRAM;
    log_info(
        tt::LogType::LogAlways, "convert_to_hwc: Starting program creation, is_input_in_dram={}", is_input_in_dram);

    const uint32_t output_shard_height = output.shard_spec()->shape[0];
    const uint32_t output_shard_width = output.shard_spec()->shape[1];
    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: Output shard dimensions - height={}, width={}",
        output_shard_height,
        output_shard_width);

    const uint32_t l1_input_shard_height = is_input_in_dram ? a.logical_shape()[-2] : a.shard_spec()->shape[0];
    const uint32_t l1_input_shard_width = is_input_in_dram ? output_shard_height : a.shard_spec()->shape[1];
    const CoreRangeSet l1_input_core_grid = is_input_in_dram ? output.shard_spec()->grid : a.shard_spec()->grid;
    const std::vector<CoreCoord> l1_input_cores = corerange_to_cores(
        l1_input_core_grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: L1 input shard dimensions - height={}, width={}, num_cores={}",
        l1_input_shard_height,
        l1_input_shard_width,
        l1_input_cores.size());

    TT_FATAL(
        l1_input_shard_height <= TILE_HEIGHT, "Shard height must be 32 or smaller (was {})", l1_input_shard_height);
    TT_FATAL(
        l1_input_shard_width % TILE_WIDTH == 0,
        "Shard width must be multiple of tile width (was {})",
        l1_input_shard_width);
    const auto alignment_elements = compute_alignment_requirement_in_elements(output);
    log_info(tt::LogType::LogAlways, "convert_to_hwc: Alignment elements={}", alignment_elements);
    TT_FATAL(alignment_elements != 0, "Number of alignment elements cannot be 0");
    TT_FATAL(
        output_shard_width % alignment_elements == 0,
        "Output shard width must be multiple of {} to satisfy alignment constraints (was {})",
        alignment_elements,
        output_shard_width);

    const auto create_circular_buffer = [&program, &l1_input_core_grid](
                                            uint32_t index,
                                            uint32_t total_size,
                                            uint32_t page_size,
                                            const tt::DataFormat& format,
                                            tt::tt_metal::Buffer* buffer = nullptr) -> tt::tt_metal::CBHandle {
        log_info(
            tt::LogType::LogAlways,
            "Creating circular buffer id={} -> page_size={}, total_size={}",
            index,
            page_size,
            total_size);
        auto config = tt::tt_metal::CircularBufferConfig(total_size, {{index, format}}).set_page_size(index, page_size);
        if (buffer != nullptr) {
            config = config.set_globally_allocated_address(*buffer);
        }
        return tt::tt_metal::CreateCircularBuffer(program, l1_input_core_grid, config);
    };

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_element_size = tt::datum_size(input_format);
    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: Input format={}, element_size={}",
        static_cast<int>(input_format),
        input_element_size);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);
    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: Intermediary format={}, tile_size={}",
        static_cast<int>(intermediary_format),
        intermediary_tile_size);

    const uint32_t cb_full_input_id = tt::CBIndex::c_5;
    const uint32_t cb_full_input_page_size = l1_input_shard_width * input_element_size;
    const uint32_t cb_full_input_total_size = l1_input_shard_height * cb_full_input_page_size;
    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: CB full input - id={}, page_size={}, total_size={}",
        cb_full_input_id,
        cb_full_input_page_size,
        cb_full_input_total_size);
    const auto cb_full_input = create_circular_buffer(
        cb_full_input_id,
        cb_full_input_total_size,
        cb_full_input_page_size,
        input_format,
        is_input_in_dram ? nullptr : a.buffer());  // DONT DO THIS IF DRAM

    const uint32_t cb_in_id = tt::CBIndex::c_0;
    const uint32_t cb_in_page_size = l1_input_shard_width * input_element_size;
    const uint32_t cb_in_total_size = l1_input_shard_height * cb_in_page_size;
    create_circular_buffer(cb_in_id, cb_in_total_size, cb_in_page_size, input_format);

    const uint32_t cb_in_tiled_id = tt::CBIndex::c_1;
    const uint32_t cb_in_tiled_total_size = tt::div_up(l1_input_shard_width, TILE_WIDTH) * intermediary_tile_size;
    const uint32_t cb_in_tiled_page_size = intermediary_tile_size;
    create_circular_buffer(cb_in_tiled_id, cb_in_tiled_total_size, cb_in_tiled_page_size, intermediary_format);

    const uint32_t cb_in_transpose_total_size = tt::div_up(l1_input_shard_width, TILE_WIDTH) * intermediary_tile_size;
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;

    const uint32_t cb_in_transpose_id0 = tt::CBIndex::c_2;
    create_circular_buffer(
        cb_in_transpose_id0, cb_in_transpose_total_size, cb_in_transpose_page_size, intermediary_format);

    const uint32_t cb_in_transpose_id1 = tt::CBIndex::c_3;
    create_circular_buffer(
        cb_in_transpose_id1, cb_in_transpose_total_size, cb_in_transpose_page_size, intermediary_format);

    const uint32_t cb_out_id = tt::CBIndex::c_4;
    const tt::DataFormat output_format = input_format;
    const uint32_t cb_out_total_size = cb_in_total_size;  // same size as input
    const uint32_t cb_out_page_size = l1_input_shard_height * input_element_size;
    const auto cb_out =
        create_circular_buffer(cb_out_id, cb_out_total_size, cb_out_page_size, output_format, output.buffer());

    const uint32_t total_tiles_per_core = tt::div_up(l1_input_shard_width, TILE_HEIGHT);
    const uint32_t total_tiles_writer0 = tt::div_up(total_tiles_per_core, 2);
    const uint32_t total_tiles_writer1 = total_tiles_per_core - total_tiles_writer0;
    uint32_t output_stride_sticks = TILE_WIDTH;  // needed to stride output address when doing split writers
    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: Tiles - total_per_core={}, writer0={}, writer1={}, output_stride_sticks={}",
        total_tiles_per_core,
        total_tiles_writer0,
        total_tiles_writer1,
        output_stride_sticks);

    const auto dram_input_cores = corerange_to_cores(
        a.shard_spec()->grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    const auto remote_core_type = a.buffer()->core_type();
    const auto remote_address = a.buffer()->address();
    const auto remote_buffer_type = a.buffer()->buffer_type();
    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: DRAM config - num_dram_cores={}, remote_address=0x{:x}, core_type={}, buffer_type={}",
        dram_input_cores.size(),
        remote_address,
        static_cast<int>(remote_core_type),
        static_cast<int>(remote_buffer_type));
    const auto [runtime_args_for_each_core, _, dram_write_stride_bytes, dram_read_stride_bytes] =
        data_movement::detail::compute_width_sharding_reshard_segments(
            {l1_input_shard_height, l1_input_shard_width},
            a.shard_spec()->shape,  // dram shard shape
            l1_input_cores,
            dram_input_cores,
            remote_buffer_type,
            remote_core_type,
            a.device(),
            2);
    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: Stride bytes - dram_write_stride_bytes={}, dram_read_stride_bytes={}",
        dram_write_stride_bytes,
        dram_read_stride_bytes);
    log_info(tt::LogType::LogAlways, "runtime_args_for_each_core[0]={}", runtime_args_for_each_core[0]);

    // Split DRAM read across each kernel along tensor height since this is the best way to split work evenly
    const uint32_t total_num_sticks_kernel_0 = l1_input_shard_height / 2;
    const uint32_t total_num_sticks_kernel_1 = l1_input_shard_height - total_num_sticks_kernel_0;

    const uint32_t batch_size = 1;  // TODO: get this from the right place
    const uint32_t num_sticks_block_size_kernel_0 = total_num_sticks_kernel_0 / batch_size;
    const uint32_t num_sticks_block_size_kernel_1 = total_num_sticks_kernel_1 / batch_size;

    log_info(
        tt::LogType::LogAlways,
        "convert_to_hwc: Kernel work split - kernel0_sticks={} (block_size={}), kernel1_sticks={} (block_size={})",
        total_num_sticks_kernel_0,
        num_sticks_block_size_kernel_0,
        total_num_sticks_kernel_1,
        num_sticks_block_size_kernel_1);

    std::vector<uint32_t> writer_compile_time_args0 = {
        cb_full_input_id,
        cb_in_id,
        cb_in_transpose_id0,
        cb_out_id,
        output_shard_width,   // output channels
        output_shard_height,  // output hw
        total_tiles_writer0,
        output_stride_sticks,
        0,
        input_element_size,
        is_input_in_dram,
        true,
        dram_write_stride_bytes,
        dram_read_stride_bytes,
        total_num_sticks_kernel_0,
        num_sticks_block_size_kernel_0,
        batch_size};
    log_info(tt::LogType::LogAlways, "convert_to_hwc: writer_compile_time_args0 = {}", writer_compile_time_args0);

    std::vector<uint32_t> writer_compile_time_args1 = {
        cb_full_input_id,
        cb_in_id,
        cb_in_transpose_id1,
        cb_out_id,
        output_shard_width,   // output channels
        output_shard_height,  // output hw
        total_tiles_writer1,
        output_stride_sticks,
        output_stride_sticks,
        input_element_size,
        is_input_in_dram,
        false,
        dram_write_stride_bytes,
        dram_read_stride_bytes,
        total_num_sticks_kernel_1,
        num_sticks_block_size_kernel_1,
        batch_size};
    log_info(tt::LogType::LogAlways, "convert_to_hwc: writer_compile_time_args1 = {}", writer_compile_time_args1);

    std::vector<uint32_t> compute_compile_time_args = {
        cb_in_id,
        cb_in_tiled_id,
        cb_in_transpose_id0,
        cb_in_transpose_id1,
        total_tiles_per_core,
        l1_input_shard_height,
        is_input_in_dram};
    log_info(tt::LogType::LogAlways, "convert_to_hwc: compute_compile_time_args = {}", compute_compile_time_args);

    log_info(tt::LogType::LogAlways, "convert_to_hwc: Creating kernels on {} cores", l1_input_cores.size());
    auto writer_kernel_id0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        l1_input_core_grid,
        tt::tt_metal::ReaderDataMovementConfig(writer_compile_time_args0));

    auto writer_kernel_id1 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        l1_input_core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args1));

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/convert_to_hwc.cpp",
        l1_input_core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    log_info(tt::LogType::LogAlways, "convert_to_hwc: Kernels created successfully");

    auto set_runtime_args = [cb_full_input,
                             cb_out,
                             is_input_in_dram,
                             l1_input_cores,
                             runtime_args_for_each_core,
                             writer_kernel_id0,
                             writer_kernel_id1,
                             total_num_sticks_kernel_0,
                             total_num_sticks_kernel_1,
                             remote_address,
                             dram_read_stride_bytes,
                             dram_write_stride_bytes](
                                tt::tt_metal::Program& program, const Tensor& a, const Tensor& output) {
        log_info(tt::LogType::LogAlways, "convert_to_hwc: Setting runtime args, is_input_in_dram={}", is_input_in_dram);

        for (uint32_t core_idx = 0; core_idx < l1_input_cores.size(); core_idx++) {
            const auto& args_for_all_segments = runtime_args_for_each_core[core_idx];
            std::vector<uint32_t> runtime_args_0 = {remote_address, args_for_all_segments.size()};
            std::vector<uint32_t> runtime_args_1 = {remote_address, args_for_all_segments.size()};
            for (const auto& args : args_for_all_segments) {
                const std::vector<uint32_t> segment_kernel_0 = {
                    args.write_size, args.read_offset, args.bank_id, args.write_offset};
                runtime_args_0.insert(runtime_args_0.end(), segment_kernel_0.begin(), segment_kernel_0.end());

                // Adjust read and write offsets to the correct stick address because we are splitting work across 2
                // kernels
                const uint32_t adjusted_read_offset =
                    args.read_offset + (total_num_sticks_kernel_0 * dram_write_stride_bytes);
                const uint32_t adjusted_write_offset =
                    args.write_offset + (total_num_sticks_kernel_0 * dram_read_stride_bytes);

                const std::vector<uint32_t> segment_kernel_1 = {
                    args.write_size, adjusted_read_offset, args.bank_id, adjusted_write_offset};
                runtime_args_1.insert(runtime_args_1.end(), segment_kernel_1.begin(), segment_kernel_1.end());
            }
            log_info(
                tt::LogType::LogAlways,
                "convert_to_hwc: Core {} runtime args - writer0_size={}, writer1_size={}",
                core_idx,
                runtime_args_0.size(),
                runtime_args_1.size());

            log_info(tt::LogType::LogAlways, "runtime_args_0={}\n runtime_args_1={}", runtime_args_0, runtime_args_1);
            SetRuntimeArgs(program, writer_kernel_id0, l1_input_cores[core_idx], runtime_args_0);
            SetRuntimeArgs(program, writer_kernel_id1, l1_input_cores[core_idx], runtime_args_1);
        }
        if (is_input_in_dram) {
        } else {
            tt::tt_metal::Buffer* a_buffer = a.buffer();
            log_info(tt::LogType::LogAlways, "convert_to_hwc: Updating CB addresses for L1 input");
            UpdateDynamicCircularBufferAddress(program, cb_full_input, *a_buffer);
        }
        tt::tt_metal::Buffer* output_buffer = output.buffer();
        UpdateDynamicCircularBufferAddress(program, cb_out, *output_buffer);
        log_info(tt::LogType::LogAlways, "convert_to_hwc: Runtime args setup complete");
    };
    set_runtime_args(program, a, output);

    auto override_runtime_arguments_callback = [set_runtime_args](
                                                   const void* operation,
                                                   tt::tt_metal::Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& output_tensor = output_tensors.at(0);
        set_runtime_args(program, input_tensors.at(0), output_tensor);
    };

    log_info(tt::LogType::LogAlways, "convert_to_hwc: Program creation complete");
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::cnn::detail
