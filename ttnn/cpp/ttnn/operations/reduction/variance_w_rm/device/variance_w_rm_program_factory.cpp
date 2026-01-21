// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "variance_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::reduction::variance_w_rm::program {

using namespace tt;
using namespace tt::tt_metal;

// Factory implementation for Stages 5-6
VarianceWRmProgramFactory::cached_program_t VarianceWRmProgramFactory::create(
    const VarianceWRmParams& operation_attributes, const VarianceWRmInputs& tensor_args, Tensor& tensor_return_value) {
    (void)operation_attributes;  // Unused in single-core implementation

    // ============================================================
    // CONSTANTS - Define buffering factor
    // ============================================================
    constexpr uint32_t buffering_factor = 2;  // Double buffering

    // ============================================================
    // 1. Extract tensor properties (ALL const)
    // ============================================================
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto* src_buffer = input.buffer();
    const auto* dst_buffer = output.buffer();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(cb_data_format);

    // Tensor dimensions (const)
    const uint32_t W = input.padded_shape()[-1];
    const uint32_t H = input.padded_shape()[-2];
    const uint32_t Wt = W / tt::constants::TILE_WIDTH;
    const uint32_t Ht = H / tt::constants::TILE_HEIGHT;

    // Stick sizes (unaligned and aligned for NoC) - will be used in Stage 6
    const uint32_t input_stick_size = W * input.element_size();
    const uint32_t input_stick_size_aligned = tt::round_up(input_stick_size, src_buffer->alignment());
    const uint32_t output_stick_size = 32 * output.element_size();  // Output width = 32 (reduced to 1 tile)
    const uint32_t output_stick_size_aligned = tt::round_up(output_stick_size, dst_buffer->alignment());

    // Suppress unused warnings for Stage 5 (used in Stage 6 for kernels)
    (void)Ht;
    (void)input_stick_size_aligned;
    (void)output_stick_size_aligned;

    // ============================================================
    // 2. Create program
    // ============================================================
    Program program = Program();

    // ============================================================
    // 3. Work distribution - Single core for initial implementation
    // ============================================================
    const CoreCoord single_core = {0, 0};
    const CoreRange single_core_range(single_core, single_core);
    const CoreRangeSet all_cores(single_core_range);

    // ============================================================
    // 4. Create circular buffers (8 CBs as per spec)
    //    - CB c_0: Input row-major sticks (Wt tiles per tile-row)
    //    - CB c_1: Tiled input (Wt tiles, must persist for bcast_sub)
    //    - CB c_2: Scaler tile (1 tile, persistent, contains 1/W)
    //    - CB c_3: Mean tiles (1 tile)
    //    - CB c_4: Centralized tiles (Wt tiles)
    //    - CB c_5: Squared tiles (Wt tiles)
    //    - CB c_6: Variance tile (1 tile)
    //    - CB c_16: Output row-major sticks (1 tile = 32 sticks of width 32)
    // ============================================================

    // CB c_0: Input RM sticks (double-buffered, Wt tiles worth per tile-row)
    constexpr uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
    const uint32_t cb_in_rm_page_size = tile_size;              // Tile-sized for tilize sync
    const uint32_t cb_in_rm_num_pages = buffering_factor * Wt;  // 2 * Wt for double buffering
    tt::tt_metal::create_cb(cb_in_rm_idx, program, all_cores, cb_in_rm_page_size, cb_in_rm_num_pages, cb_data_format);

    // CB c_1: Tiled input (MUST hold Wt tiles - needed for bcast_sub after reduce)
    constexpr uint32_t cb_in_tiled_idx = tt::CBIndex::c_1;
    const uint32_t cb_in_tiled_page_size = tile_size;
    const uint32_t cb_in_tiled_num_pages = Wt;  // Hold full tile-row for subtraction
    tt::tt_metal::create_cb(
        cb_in_tiled_idx, program, all_cores, cb_in_tiled_page_size, cb_in_tiled_num_pages, cb_data_format);

    // CB c_2: Scaler tile (1 tile, persistent, contains 1/W for both reduces)
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    const tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;  // bfloat16 for scaler
    const uint32_t scaler_tile_size = tt::tile_size(scaler_cb_data_format);
    const uint32_t cb_scaler_page_size = scaler_tile_size;
    const uint32_t cb_scaler_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_scaler_idx, program, all_cores, cb_scaler_page_size, cb_scaler_num_pages, scaler_cb_data_format);

    // CB c_3: Mean tiles (1 tile, output width = 1 tile after first reduction)
    constexpr uint32_t cb_mean_tiled_idx = tt::CBIndex::c_3;
    const uint32_t cb_mean_tiled_page_size = tile_size;
    const uint32_t cb_mean_tiled_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_mean_tiled_idx, program, all_cores, cb_mean_tiled_page_size, cb_mean_tiled_num_pages, cb_data_format);

    // CB c_4: Centralized tiled data (Wt tiles)
    constexpr uint32_t cb_centralized_tiled_idx = tt::CBIndex::c_4;
    const uint32_t cb_centralized_tiled_page_size = tile_size;
    const uint32_t cb_centralized_tiled_num_pages = Wt;  // Full tile-row width
    tt::tt_metal::create_cb(
        cb_centralized_tiled_idx,
        program,
        all_cores,
        cb_centralized_tiled_page_size,
        cb_centralized_tiled_num_pages,
        cb_data_format);

    // CB c_5: Squared tiles (Wt tiles)
    constexpr uint32_t cb_squared_tiled_idx = tt::CBIndex::c_5;
    const uint32_t cb_squared_tiled_page_size = tile_size;
    const uint32_t cb_squared_tiled_num_pages = Wt;  // Full tile-row width
    tt::tt_metal::create_cb(
        cb_squared_tiled_idx,
        program,
        all_cores,
        cb_squared_tiled_page_size,
        cb_squared_tiled_num_pages,
        cb_data_format);

    // CB c_6: Variance tile (1 tile after second reduction)
    constexpr uint32_t cb_variance_tiled_idx = tt::CBIndex::c_6;
    const uint32_t cb_variance_tiled_page_size = tile_size;
    const uint32_t cb_variance_tiled_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_variance_tiled_idx,
        program,
        all_cores,
        cb_variance_tiled_page_size,
        cb_variance_tiled_num_pages,
        cb_data_format);

    // CB c_16: Output after untilize (1 tile = 32 sticks of width 32, double-buffered)
    constexpr uint32_t cb_out_rm_idx = tt::CBIndex::c_16;
    const uint32_t cb_out_rm_page_size = tile_size;         // Tile-sized for untilize helper
    const uint32_t cb_out_rm_num_pages = buffering_factor;  // Only 1 tile output per tile-row, double-buffered
    tt::tt_metal::create_cb(
        cb_out_rm_idx, program, all_cores, cb_out_rm_page_size, cb_out_rm_num_pages, cb_data_format);

    // ============================================================
    // 5. Create kernels
    // ============================================================

    // Calculate scaler value: 1/W for mean calculation
    const float scaler_value = 1.0f / static_cast<float>(W);
    bfloat16 bfloat_scaler_value = bfloat16::truncate(scaler_value);
    const uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

    // Reader compile-time args: input_stick_size, packed_scaler, Ht, Wt, TensorAccessorArgs
    std::vector<uint32_t> reader_compile_time_args = {input_stick_size_aligned, packed_scaler_value, Ht, Wt};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    // Compute compile-time args: Ht, Wt
    const std::vector<uint32_t> compute_compile_time_args = {Ht, Wt};

    // Writer compile-time args: output_stick_size, Ht, TensorAccessorArgs
    std::vector<uint32_t> writer_compile_time_args = {output_stick_size_aligned, Ht};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Reader kernel (RISCV_0 / BRISC / NOC0)
    const auto reader_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/variance_w_rm/device/kernels/dataflow/reader_variance_w_rm.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Compute kernel (RISCV_2,3,4 / Unpack, Math, Pack)
    const auto compute_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/variance_w_rm/device/kernels/compute/variance_w_rm_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

    // Writer kernel (RISCV_1 / NCRISC / NOC1)
    const auto writer_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/variance_w_rm/device/kernels/dataflow/writer_variance_w_rm.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // ============================================================
    // 6. Set runtime args (minimal for stubs)
    // ============================================================
    const std::vector<uint32_t> reader_runtime_args = {src_buffer->address()};
    const std::vector<uint32_t> compute_runtime_args = {};
    const std::vector<uint32_t> writer_runtime_args = {dst_buffer->address()};

    SetRuntimeArgs(program, reader_id, single_core, reader_runtime_args);
    SetRuntimeArgs(program, compute_id, single_core, compute_runtime_args);
    SetRuntimeArgs(program, writer_id, single_core, writer_runtime_args);

    // ============================================================
    // 7. Return cached program with shared variables
    // ============================================================
    return {
        std::move(program),
        VarianceWRmSharedVariables{
            .reader_kernel_id = reader_id,
            .compute_kernel_id = compute_id,
            .writer_kernel_id = writer_id,
            .all_cores = all_cores,
            .num_cores = 1}};
}

void VarianceWRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const VarianceWRmParams& operation_attributes,
    const VarianceWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    (void)operation_attributes;  // Unused in single-core implementation

    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    // Update buffer addresses for cached program reuse
    const uint32_t input_addr = input.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();

    // Single core - update runtime args directly
    const CoreCoord single_core = {0, 0};
    SetRuntimeArgs(program, reader_kernel_id, single_core, {input_addr});
    SetRuntimeArgs(program, writer_kernel_id, single_core, {output_addr});
}

}  // namespace ttnn::operations::reduction::variance_w_rm::program
