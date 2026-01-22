// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "standardize_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
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
    const uint32_t W = input.logical_shape()[-1];                                         // Width in elements
    const uint32_t Wt = (W + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;  // Width in tiles

    // Suppress unused variable warnings (epsilon, Ht will be used in Stage 6)
    (void)operation_attributes.epsilon;

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

    // CB c_0: Input RM sticks (double-buffered)
    constexpr uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
    const uint32_t cb_in_rm_page_size = input_stick_size_aligned;
    const uint32_t cb_in_rm_num_pages = buffering_factor * tt::constants::TILE_HEIGHT;  // 2 * 32 = 64 sticks
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
    constexpr uint32_t cb_out_rm_idx = tt::CBIndex::c_16;
    const uint32_t cb_out_rm_page_size = output_stick_size_aligned;
    const uint32_t cb_out_rm_num_pages = buffering_factor * tt::constants::TILE_HEIGHT;  // 2 * 32 = 64 sticks
    tt::tt_metal::create_cb(
        cb_out_rm_idx, program, core_range_set, cb_out_rm_page_size, cb_out_rm_num_pages, cb_data_format);

    // ============================================================
    // 5. Create kernels (empty stubs for Stage 6)
    // ============================================================

    // Reader kernel (empty stub)
    const std::vector<uint32_t> reader_compile_args = {};
    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/dataflow/reader_standardize_w_rm.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // Compute kernel (empty stub)
    const std::vector<uint32_t> compute_compile_args = {};
    const auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/compute/standardize_w_rm_compute.cpp",
        core_range_set,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_args});

    // Writer kernel (empty stub)
    const std::vector<uint32_t> writer_compile_args = {};
    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/dataflow/writer_standardize_w_rm.cpp",
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // ============================================================
    // 6. Set runtime args (empty for stub kernels)
    // ============================================================
    const uint32_t input_addr = src_buffer->address();
    const uint32_t output_addr = dst_buffer->address();

    SetRuntimeArgs(program, reader_kernel_id, core, {input_addr});
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
