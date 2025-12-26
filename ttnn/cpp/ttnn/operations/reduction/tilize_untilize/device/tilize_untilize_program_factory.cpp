// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_untilize_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "kernels/op_types.hpp"

namespace ttnn::operations::reduction::program {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

TilizeUntilizeProgramFactory::cached_program_t TilizeUntilizeProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;

    tt::tt_metal::Program program{};

    // Get buffers
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t input_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_dest_acc_en = input.dtype() == DataType::FLOAT32;

    // Get tensor dimensions
    // Shape is [N, C, H, W]
    auto shape = input.padded_shape();
    uint32_t tensor_height = shape[-2];
    uint32_t tensor_width = shape[-1];

    // Calculate tiles
    uint32_t num_tiles_per_row = tensor_width / TILE_WIDTH;
    uint32_t num_tile_rows = tensor_height / TILE_HEIGHT;

    // Calculate batch dimensions (N * C)
    uint32_t batch_size = 1;
    if (shape.rank() >= 3) {
        batch_size = shape[-3];
    }
    if (shape.rank() >= 4) {
        batch_size *= shape[-4];
    }

    // Total number of tile blocks (each block = 1 tile row = 32 rows × width)
    uint32_t num_blocks = num_tile_rows * batch_size;

    // Stick (row) dimensions
    uint32_t stick_size = tensor_width * input.element_size();

    // For reduction operations, output has width of TILE_WIDTH (32)
    uint32_t output_stick_size = stick_size;  // Same as input for IDENTITY
    if (operation_attributes.op_type != OpType::IDENTITY) {
        // Reductions produce 1 tile per row (width = 32)
        output_stick_size = TILE_WIDTH * input.element_size();
    }

    // Device and core setup - multi-core work distribution
    IDevice* device = input.device();
    auto grid_size = device->compute_with_storage_grid_size();

    // Split work across cores using height-based 1D parallelization
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, num_blocks);

    bool has_cliff = !core_range_cliff.empty();
    auto cores = corerange_to_cores(all_cores);

    // ============================================================
    // CIRCULAR BUFFER CREATION
    // Double-buffered for multi-core pipelining when processing 2+ blocks
    // CB_in (c_0): Row-major input staging
    // CB_tiled (c_1): Tiled intermediate (single-buffered, tilize->untilize is atomic)
    // CB_out (c_16): Row-major output staging
    // ============================================================

    // Double-buffer CB_in and CB_out when processing multiple blocks for pipelining
    uint32_t cb_num_tiles = (nblocks_per_core > 1) ? num_tiles_per_row * 2 : num_tiles_per_row;

    // CB_in (c_0): Input circular buffer for row-major data from reader
    uint32_t cb_in_id = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in_config =
        tt::tt_metal::CircularBufferConfig(cb_num_tiles * input_tile_size, {{cb_in_id, input_cb_data_format}})
            .set_page_size(cb_in_id, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_config);

    // CB_tiled (c_1): Intermediate circular buffer for tiled data
    // Single-buffered - tilize->untilize happens atomically per block
    uint32_t cb_tiled_id = tt::CBIndex::c_1;
    uint32_t cb_tiled_num_tiles = num_tiles_per_row;
    tt::tt_metal::CircularBufferConfig cb_tiled_config =
        tt::tt_metal::CircularBufferConfig(cb_tiled_num_tiles * input_tile_size, {{cb_tiled_id, input_cb_data_format}})
            .set_page_size(cb_tiled_id, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_tiled_config);

    // CB_out (c_16): Output circular buffer for row-major data to writer
    uint32_t cb_out_id = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(cb_num_tiles * output_tile_size, {{cb_out_id, output_cb_data_format}})
            .set_page_size(cb_out_id, output_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // ============================================================
    // KERNEL CREATION
    // ============================================================

    // Operation type from attributes - no recompilation needed to change
    uint32_t op_type = static_cast<uint32_t>(operation_attributes.op_type);

    // Compute scaler: use provided scaler, with special handling for AVG
    float scaler_value = operation_attributes.scaler;
    if (operation_attributes.op_type == OpType::REDUCE_W_AVG) {
        // For AVG, scaler should be 1/W. If user provided 1.0f (default),
        // compute it automatically. Otherwise, use user-provided value.
        if (scaler_value == 1.0f) {
            scaler_value = 1.0f / static_cast<float>(tensor_width);
        }
    }
    bfloat16 bf_scaler = bfloat16::truncate(scaler_value);
    uint32_t packed_scaler = pack_two_bfloat16_into_uint32({bf_scaler, bf_scaler});

    // Create CB_scaler (c_2) and CB_reduced (c_3) for reduction operations
    if (op_type != static_cast<uint32_t>(OpType::IDENTITY)) {
        // CB_scaler (c_2): Scaler tile for reduction operations
        // Only 1 tile needed - generated once by reader, never popped (persistent read)
        tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
        uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
        tt::tt_metal::CircularBufferConfig cb_scaler_config =
            tt::tt_metal::CircularBufferConfig(1 * scaler_single_tile_size, {{tt::CBIndex::c_2, scaler_cb_data_format}})
                .set_page_size(tt::CBIndex::c_2, scaler_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

        // CB_reduced (c_3): Holds the reduced tile between reduce and untilize
        // Data flow: CB_1 (num_tiles) -> reduce -> CB_3 (1 tile) -> untilize -> CB_16
        // Single-buffered: same kernel writes (reduce) and reads (untilize), no overlap possible
        tt::tt_metal::CircularBufferConfig cb_reduced_config =
            tt::tt_metal::CircularBufferConfig(input_tile_size, {{tt::CBIndex::c_3, input_cb_data_format}})
                .set_page_size(tt::CBIndex::c_3, input_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_reduced_config);
    }

    // Generate kernel defines for reduction operations
    std::map<std::string, std::string> kernel_defines;
    if (op_type == static_cast<uint32_t>(OpType::REDUCE_W_SUM) ||
        op_type == static_cast<uint32_t>(OpType::REDUCE_W_AVG)) {
        kernel_defines["REDUCE_OP"] = "PoolType::SUM";
        kernel_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    } else if (op_type == static_cast<uint32_t>(OpType::REDUCE_W_MAX)) {
        kernel_defines["REDUCE_OP"] = "PoolType::MAX";
        kernel_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    }

    // Compile-time args for reader kernel
    // Layout: [stick_size, op_type, packed_scaler, TensorAccessorArgs...]
    std::vector<uint32_t> reader_compile_time_args = {stick_size, op_type, packed_scaler};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    // Compile-time args for writer kernel
    // Use output_stick_size (different from input for reductions)
    std::vector<uint32_t> writer_compile_time_args = {cb_out_id, output_stick_size, TILE_HEIGHT};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Compile-time args for compute kernel
    // Layout: [num_tiles_per_row, op_type]
    // num_blocks is runtime (varies per core to handle cliff)
    std::vector<uint32_t> compute_compile_time_args = {num_tiles_per_row, op_type};

    // Create reader kernel (RISCV_0 / BRISC / NOC0)
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/dataflow/"
        "reader_tilize_untilize_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    // Create writer kernel (RISCV_1 / NCRISC / NOC1)
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/dataflow/"
        "writer_tilize_untilize_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel - single kernel handles all cores (full + cliff)
    // num_blocks passed as runtime arg to handle varying block counts
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/compute/tilize_untilize_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_time_args,
            .defines = kernel_defines});

    // ============================================================
    // SET PER-CORE RUNTIME ARGUMENTS
    // ============================================================

    uint32_t row_start_id = 0;

    for (uint32_t i = 0; i < ncores; ++i) {
        const CoreCoord& core = cores[i];

        // Determine block count for this core (cliff core gets remainder)
        bool is_cliff_core = has_cliff && (i == ncores - 1);
        uint32_t blocks_for_this_core = is_cliff_core ? nblocks_per_core_cliff : nblocks_per_core;
        uint32_t sticks_for_this_core = blocks_for_this_core * TILE_HEIGHT;

        // Reader: src_addr, num_sticks, start_stick_id
        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, {src_buffer->address(), sticks_for_this_core, row_start_id});

        // Writer: dst_addr, num_blocks, start_stick_id
        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {dst_buffer->address(), blocks_for_this_core, row_start_id});

        // Compute: num_blocks (runtime arg to handle cliff)
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {blocks_for_this_core});

        row_start_id += sticks_for_this_core;
    }

    return {
        std::move(program),
        TilizeUntilizeSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .all_cores = all_cores,
            .cores = cores,
            .num_cores = ncores}};
}

void TilizeUntilizeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    const auto& input = tensor_args.input;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Update buffer addresses for all cores
    // Only the address (index 0) needs updating; other args are unchanged
    for (const auto& core : shared_vars.cores) {
        {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::reduction::program
