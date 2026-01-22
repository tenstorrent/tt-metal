// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "standardize_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::reduction::standardize_w_rm::program {

using namespace tt;
using namespace tt::tt_metal;

StandardizeWRmProgramFactory::cached_program_t StandardizeWRmProgramFactory::create(
    const StandardizeWRmParams& operation_attributes,
    const StandardizeWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    // ============================================================
    // CONSTANTS
    // ============================================================
    constexpr uint32_t buffering_factor = 2;  // Double buffering

    // ============================================================
    // 1. Extract tensor properties
    // ============================================================
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    const auto* src_buffer = input.buffer();
    const auto* dst_buffer = output.buffer();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(cb_data_format);

    // Tensor dimensions (ROW_MAJOR layout)
    // Use padded_shape for actual memory layout
    const uint32_t W = input.padded_shape()[-1];         // Width in elements (padded)
    const uint32_t H = input.padded_shape()[-2];         // Height in elements (padded)
    const uint32_t Wt = W / tt::constants::TILE_WIDTH;   // Width in tiles
    const uint32_t Ht = H / tt::constants::TILE_HEIGHT;  // Height in tiles

    // Stick sizes (ROW_MAJOR)
    const uint32_t input_stick_size = W * input.element_size();
    const uint32_t input_stick_size_aligned = tt::round_up(input_stick_size, src_buffer->alignment());
    const uint32_t output_stick_size = W * output.element_size();
    const uint32_t output_stick_size_aligned = tt::round_up(output_stick_size, dst_buffer->alignment());

    // ============================================================
    // 2. Create program
    // ============================================================
    Program program = Program();

    // ============================================================
    // 3. Work distribution (single core for now)
    // ============================================================
    const CoreCoord core = {0, 0};
    const CoreRange all_cores(core, core);
    const CoreRangeSet core_range_set({all_cores});

    // ============================================================
    // 4. Create circular buffers (10 CBs total)
    // ============================================================

    // CB c_0: Input RM sticks
    // Page size = tile_size for tilize sync (helper expects Wt pages)
    // Num pages = 2*Wt for double-buffering one tile-row
    constexpr uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
    const uint32_t cb_in_rm_page_size = tile_size;
    const uint32_t cb_in_rm_num_pages = buffering_factor * Wt;
    tt::tt_metal::create_cb(
        cb_in_rm_idx, program, core_range_set, cb_in_rm_page_size, cb_in_rm_num_pages, cb_data_format);

    // CB c_1: Tiled input (PERSISTENT for reduce + subtract)
    constexpr uint32_t cb_in_tiled_idx = tt::CBIndex::c_1;
    const uint32_t cb_in_tiled_page_size = tile_size;
    const uint32_t cb_in_tiled_num_pages = Wt;  // Hold one tile-row
    tt::tt_metal::create_cb(
        cb_in_tiled_idx, program, core_range_set, cb_in_tiled_page_size, cb_in_tiled_num_pages, cb_data_format);

    // CB c_2: Scaler (1/W) for reduces
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    const uint32_t cb_scaler_page_size = tile_size;
    const uint32_t cb_scaler_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_scaler_idx, program, core_range_set, cb_scaler_page_size, cb_scaler_num_pages, cb_data_format);

    // CB c_3: Mean tile
    constexpr uint32_t cb_mean_idx = tt::CBIndex::c_3;
    const uint32_t cb_mean_page_size = tile_size;
    const uint32_t cb_mean_num_pages = 1;
    tt::tt_metal::create_cb(cb_mean_idx, program, core_range_set, cb_mean_page_size, cb_mean_num_pages, cb_data_format);

    // CB c_4: Centralized tiles (PERSISTENT through phases 3-8)
    constexpr uint32_t cb_centralized_idx = tt::CBIndex::c_4;
    const uint32_t cb_centralized_page_size = tile_size;
    const uint32_t cb_centralized_num_pages = Wt;  // Hold full tile-row
    tt::tt_metal::create_cb(
        cb_centralized_idx,
        program,
        core_range_set,
        cb_centralized_page_size,
        cb_centralized_num_pages,
        cb_data_format);

    // CB c_5: Squared tiles
    constexpr uint32_t cb_squared_idx = tt::CBIndex::c_5;
    const uint32_t cb_squared_page_size = tile_size;
    const uint32_t cb_squared_num_pages = Wt;  // Hold full tile-row
    tt::tt_metal::create_cb(
        cb_squared_idx, program, core_range_set, cb_squared_page_size, cb_squared_num_pages, cb_data_format);

    // CB c_6: Variance tile
    constexpr uint32_t cb_variance_idx = tt::CBIndex::c_6;
    const uint32_t cb_variance_page_size = tile_size;
    const uint32_t cb_variance_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_variance_idx, program, core_range_set, cb_variance_page_size, cb_variance_num_pages, cb_data_format);

    // CB c_7: Epsilon scalar tile
    constexpr uint32_t cb_epsilon_idx = tt::CBIndex::c_7;
    const uint32_t cb_epsilon_page_size = tile_size;
    const uint32_t cb_epsilon_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_epsilon_idx, program, core_range_set, cb_epsilon_page_size, cb_epsilon_num_pages, cb_data_format);

    // CB c_8: Rsqrt result tile
    constexpr uint32_t cb_rsqrt_idx = tt::CBIndex::c_8;
    const uint32_t cb_rsqrt_page_size = tile_size;
    const uint32_t cb_rsqrt_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_rsqrt_idx, program, core_range_set, cb_rsqrt_page_size, cb_rsqrt_num_pages, cb_data_format);

    // CB c_16: Output RM sticks (double-buffered)
    // Used for both Phase 8 output (tiled multiply) and Phase 9 (untilize RM output)
    // Page size = tile_size for untilize sync (helper expects Wt pages)
    // Num pages = 2*Wt for double-buffering one tile-row
    constexpr uint32_t cb_out_rm_idx = tt::CBIndex::c_16;
    const uint32_t cb_out_rm_page_size = tile_size;
    const uint32_t cb_out_rm_num_pages = buffering_factor * Wt;
    tt::tt_metal::create_cb(
        cb_out_rm_idx, program, core_range_set, cb_out_rm_page_size, cb_out_rm_num_pages, cb_data_format);

    // ============================================================
    // 5. Create kernels with proper compile-time args
    // ============================================================

    // Calculate scaler value: 1/W for mean calculation
    const float scaler_value = 1.0f / static_cast<float>(W);
    bfloat16 bfloat_scaler_value = bfloat16::truncate(scaler_value);
    const uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

    // Pack epsilon value for numerical stability in rsqrt
    bfloat16 bfloat_epsilon_value = bfloat16::truncate(operation_attributes.epsilon);
    const uint32_t packed_epsilon_value = pack_two_bfloat16_into_uint32({bfloat_epsilon_value, bfloat_epsilon_value});

    // Reader compile-time args: input_stick_size, packed_scaler, packed_epsilon, Ht, Wt, TensorAccessorArgs
    std::vector<uint32_t> reader_compile_args = {
        input_stick_size_aligned, packed_scaler_value, packed_epsilon_value, Ht, Wt};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_args);

    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/dataflow/reader_standardize_w_rm.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // Compute compile-time args: Ht, Wt
    const std::vector<uint32_t> compute_compile_args = {Ht, Wt};
    const auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/compute/standardize_w_rm_compute.cpp",
        core_range_set,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_args});

    // Writer compile-time args: output_stick_size, Ht, Wt, TensorAccessorArgs
    std::vector<uint32_t> writer_compile_args = {output_stick_size_aligned, Ht, Wt};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_args);

    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/dataflow/writer_standardize_w_rm.cpp",
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // ============================================================
    // 6. Set runtime args
    // ============================================================
    const uint32_t input_addr = src_buffer->address();
    const uint32_t output_addr = dst_buffer->address();

    // Reader: buffer address
    SetRuntimeArgs(program, reader_kernel_id, core, {input_addr});
    // Writer: buffer address
    SetRuntimeArgs(program, writer_kernel_id, core, {output_addr});

    // ============================================================
    // 7. Return cached program
    // ============================================================
    return {
        std::move(program),
        StandardizeWRmSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .all_cores = core_range_set,
            .num_cores = 1}};
}

void StandardizeWRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    [[maybe_unused]] const StandardizeWRmParams& operation_attributes,
    const StandardizeWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    const uint32_t input_addr = input.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();

    // Update runtime args for single core
    const CoreCoord core = {0, 0};
    {
        auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = input_addr;
    }
    {
        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        runtime_args[0] = output_addr;
    }
}

}  // namespace ttnn::operations::reduction::standardize_w_rm::program
