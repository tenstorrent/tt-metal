// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "fold_device_op.hpp"
namespace ttnn::operations::data_movement {

using namespace tt::constants;
using namespace tt::tt_metal;

Fold::MultiCoreDRAMFold::cached_program_t fold_multi_core_tiled_interleaved(
    const Tensor& input_tensor, const Tensor& output, const uint32_t stride_h, const uint32_t stride_w) {
    // Get device and create a new program
    auto* device = input_tensor.device();
    auto program = tt::tt_metal::CreateProgram();

    const uint32_t input_width = input_tensor.logical_shape()[2];

    // Get compute grid size and buffer pointers
    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat out_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t out_single_tile_size = tt::tile_size(out_cb_data_format);

    ttnn::Shape output_padded_shape = output.padded_shape();
    ttnn::Shape input_padded_shape = input_tensor.padded_shape();

    log_debug(tt::LogOp, "in_cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "out_cb_data_format: {}", out_cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "input_tensor_shape: {}", input_padded_shape);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);

    // Calculate memory layout parameters
    auto stick_nbytes =
        output_padded_shape[3] * tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(output.dtype()));
    uint32_t ntiles = input_tensor.physical_volume() / TILE_HW;
    uint32_t tiles_per_channel_dim = tt::div_up(input_padded_shape[-1], TILE_WIDTH);
    uint32_t tiles_per_width_dim = tt::div_up(input_padded_shape[-2], TILE_HEIGHT);
    uint32_t tiles_per_complete_row = tiles_per_width_dim * tiles_per_channel_dim;
    uint32_t num_blocks =
        std::ceil(static_cast<float>(ntiles) / (tiles_per_complete_row));  // Total number of blocks for batch * height

    uint32_t aligned_stick_nbytes = tt::align(stick_nbytes, TILE_WIDTH * tt::datum_size(out_cb_data_format));
    log_debug(
        tt::LogOp, "tiles_per_channel_dim: {}, ntiles: {}, num_blocks: {}", tiles_per_channel_dim, ntiles, num_blocks);

    // Split work across cores for parallel processing
    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, num_blocks);

    log_debug(
        tt::LogOp,
        "ncores: {}, nblocks_per_core: {}, nblocks_per_core_cliff: {}",
        ncores,
        nblocks_per_core,
        nblocks_per_core_cliff);
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = tiles_per_channel_dim;

    // Create circular buffer configurations for source and destination
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * out_single_tile_size, {{src1_cb_index, out_cb_data_format}})
            .set_page_size(src1_cb_index, out_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    // Configure compile-time arguments for reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        tiles_per_channel_dim,
        tiles_per_width_dim,
        src0_cb_index,
    };
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    // Configure compile-time arguments for writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        input_width,
        stride_h,
        stride_w,
        stick_nbytes,
        aligned_stick_nbytes,
        tiles_per_channel_dim,
        tiles_per_width_dim,
        datum_size(out_cb_data_format),
        src1_cb_index,
    };
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Create reader kernel for DRAM to circular buffer data movement
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/reader_dram2cb_tiled.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel for circular buffer to DRAM data movement
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2dram_for_tiled_input.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Configure compute kernel arguments
    std::vector<uint32_t> compute_compile_time_args = {
        nblocks_per_core * tiles_per_width_dim,
        tiles_per_channel_dim,
        src0_cb_index,
        src1_cb_index,
    };

    std::vector<uint32_t> compute_compile_time_args_cliff = {
        nblocks_per_core_cliff * tiles_per_width_dim,
        tiles_per_channel_dim,
        src0_cb_index,
        src1_cb_index,
    };

    bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
    std::string compute_kernel_name =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp";
    if (tiles_per_channel_dim > MAX_PACK_UNTILIZE_WIDTH) {
        compute_kernel_name = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
    }

    log_debug(tt::LogOp, "compute_kernel_name: {}", compute_kernel_name);

    // Create main compute kernel
    tt::tt_metal::CreateKernel(
        program,
        compute_kernel_name,
        core_range,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_time_args,
        });

    // Create cliff compute kernel if needed (for handling edge cases)
    if (!core_range_cliff.ranges().empty()) {
        tt::tt_metal::CreateKernel(
            program,
            compute_kernel_name,
            core_range_cliff,
            tt::tt_metal::ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_compile_time_args_cliff,
            });
    }

    // Calculate core distribution for work
    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        ncores_full -= 1;
        full_cores = core_range;
    }

    // Set up runtime arguments for each core
    uint32_t block_start_id = 0;
    auto ncores_x = grid_size.x;
    auto ncores_y = std::ceil(static_cast<float>(ncores) / ncores_x);
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, true);
    std::vector<CoreCoord> cores_with_rtargs;

    const uint32_t patch_size = stride_h * stride_w;         // Size of each patch
    const uint32_t output_width = input_width / stride_w;    // Output width
    // Configure runtime arguments for each core
    for (auto core : cores) {
        uint32_t curr_input_height_idx = block_start_id;
        uint32_t curr_output_height_idx = curr_input_height_idx / stride_h;
        uint32_t patch_height_offset = curr_input_height_idx % stride_h;
        uint32_t output_offset = (patch_size * curr_output_height_idx * output_width) +
                                 (patch_height_offset * stride_w);  // Total output height * width
        if (!full_cores.contains(core)) {
            continue;
        }
        std::vector<uint32_t> reader_runtime_args = {
            src0_buffer->address(),
            block_start_id,
            nblocks_per_core,
        };
        std::vector<uint32_t> writer_runtime_args = {
            dst_buffer->address(),
            block_start_id,
            nblocks_per_core,
            patch_height_offset,
            output_offset,
        };
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        block_start_id += nblocks_per_core;
        cores_with_rtargs.push_back(core);
    }

    // Handle edge case for cliff cores
    if (ncores_full < ncores) {
        uint32_t curr_input_height_idx = block_start_id;
        uint32_t curr_output_height_idx = curr_input_height_idx / stride_h;
        uint32_t patch_height_offset = curr_input_height_idx % stride_h;
        uint32_t output_offset =
            (patch_size * curr_output_height_idx * output_width) + (patch_height_offset * stride_w);
        CoreCoord core = CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x};
        std::vector<uint32_t> reader_runtime_args = {
            src0_buffer->address(),
            block_start_id,
            nblocks_per_core_cliff,
        };
        std::vector<uint32_t> writer_runtime_args = {
            dst_buffer->address(),
            block_start_id,
            nblocks_per_core_cliff,
            patch_height_offset,
            output_offset,
        };
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        cores_with_rtargs.push_back(core);
    }

    return {std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cores_with_rtargs}};
}

Fold::MultiCoreDRAMFold::cached_program_t fold_multi_core_row_major_interleaved(
    const Tensor& input_tensor, const Tensor& output, const uint32_t stride_h, const uint32_t stride_w) {
    auto* device = input_tensor.device();
    auto program = tt::tt_metal::CreateProgram();

    const uint32_t batch_size = input_tensor.logical_shape()[0];
    const uint32_t input_height = input_tensor.logical_shape()[1];
    const uint32_t input_width = input_tensor.logical_shape()[2];

    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // Calculate total input work
    uint32_t total_patches = (batch_size * input_height * input_width) / (stride_h * stride_w);

    // Get compute grid size and calculate work distribution
    auto compute_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_grid_size.x;
    uint32_t num_cores_y = compute_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    log_debug(tt::LogOp, "input_tensor_shape: {}", input_tensor.padded_shape());
    log_debug(tt::LogOp, "output_tensor_shape: {}", output.padded_shape());

    // Calculate work per core based on input dimensions
    uint32_t patches_per_core = tt::div_up(total_patches, num_cores_total);

    log_debug(
        tt::LogOp,
        "total_patches: {}, num_cores_total: {}, patches_per_core: {}",
        total_patches,
        num_cores_total,
        patches_per_core);

    // Create core ranges
    CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, true);

    // Setup circular buffers
    uint32_t cb_src0_index = tt::CBIndex::c_0;

    // Calculate buffer sizes
    uint32_t stick_nbytes = input_tensor.padded_shape()[3] * tt::datum_size(cb_data_format);
    // align to DRAM read alignment.
    uint32_t aligned_stick_nbytes = tt::align(stick_nbytes, hal::get_dram_alignment());

    log_debug(
        tt::LogOp,
        "stick_nbytes: {}, aligned_stick_nbytes: {}, dram_alignment: {}",
        stick_nbytes,
        aligned_stick_nbytes,
        hal::get_dram_alignment());

    int double_buffer = 2;
    // Create source circular buffer
    auto src_cb_config =
        CircularBufferConfig(
            double_buffer * aligned_stick_nbytes * stride_w * stride_h, {{cb_src0_index, cb_data_format}})
            .set_page_size(cb_src0_index, aligned_stick_nbytes * stride_w * stride_h);
    CreateCircularBuffer(program, all_cores, src_cb_config);

    bool is_l1_aligned = stick_nbytes == aligned_stick_nbytes;

    uint32_t cb_src1_index = tt::CBIndex::c_1;
    if (!is_l1_aligned) {
        // If not L1 aligned, use a separate circular buffer for src1
        log_debug(tt::LogOp, "Using intermediate L1 scratch buffer for src1");
        auto src1_cb_config =
            CircularBufferConfig(stick_nbytes * stride_w * stride_h, {{cb_src1_index, cb_data_format}})
                .set_page_size(cb_src1_index, stick_nbytes * stride_w * stride_h);
        CreateCircularBuffer(program, all_cores, src1_cb_config);
    }

    // Create reader kernel
    std::vector<uint32_t> compile_time_args(
        {stick_nbytes,
         cb_src0_index,
         aligned_stick_nbytes,
         stride_h,
         stride_w,
         input_width,
         patches_per_core,
         cb_src1_index,
         is_l1_aligned});
    TensorAccessorArgs(*src0_buffer).append_to(compile_time_args);
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/reader_dram2cb_for_rm_input.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args));
    // Create writer kernel
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(compile_time_args));

    // Set runtime arguments for each core

    const uint32_t output_height = input_height / stride_h;
    const uint32_t output_width = input_width / stride_w;
    const uint32_t patch_size = stride_h * stride_w;
    const uint32_t output_hw = output_height * output_width;
    uint32_t curr_patches = 0;
    std::vector<CoreCoord> cores_with_rtargs;
    uint32_t src_idx, dst_idx, src_col_offset;
    for (uint32_t i = 0; i < cores.size(); i++) {
        CoreCoord core = cores[i];
        std::vector<uint32_t> reader_runtime_args = {src0_buffer->address()};
        std::vector<uint32_t> writer_runtime_args = {dst_buffer->address()};

        if (curr_patches < total_patches) {
            uint32_t output_offset = i * patches_per_core;
            uint32_t batch_idx = output_offset / output_hw;
            uint32_t batch_offset = output_offset % output_hw;
            uint32_t out_height = batch_offset / output_width;
            uint32_t out_width = batch_offset % output_width;

            uint32_t src_batch_offset = batch_idx * output_height * output_width * patch_size;
            uint32_t src_row_offset = out_height * stride_h * input_width;
            src_col_offset = out_width * stride_w;

            src_idx = src_batch_offset + src_row_offset + src_col_offset;
            dst_idx = output_offset;
        }

        curr_patches += patches_per_core;
        reader_runtime_args.push_back(src_idx);
        reader_runtime_args.push_back(src_col_offset);
        writer_runtime_args.push_back(dst_idx);
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        cores_with_rtargs.push_back(core);
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, cores_with_rtargs}};
}

Fold::MultiCoreDRAMFold::cached_program_t Fold::MultiCoreDRAMFold::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    if (tensor_args.input_tensor.layout() == Layout::TILE) {
        log_debug(tt::LogOp, "Fold operation with DRAM tiled input");
        return fold_multi_core_tiled_interleaved(
            tensor_args.input_tensor, output_tensor, operation_attributes.stride_h, operation_attributes.stride_w);
    }
    log_debug(tt::LogOp, "Fold operation with DRAM row major input");
    return fold_multi_core_row_major_interleaved(
        tensor_args.input_tensor, output_tensor, operation_attributes.stride_h, operation_attributes.stride_w);
}

void Fold::MultiCoreDRAMFold::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& cores_with_rtargs = cached_program.shared_variables.cores_with_rtargs;

    auto& program = cached_program.program;

    const auto& input_tensor = tensor_args.input_tensor;
    auto* src_dram_buffer = input_tensor.buffer();

    auto* dst_dram_buffer = output_tensor.buffer();

    // Update runtime arguments for each core
    for (auto core : cores_with_rtargs) {
        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::data_movement
