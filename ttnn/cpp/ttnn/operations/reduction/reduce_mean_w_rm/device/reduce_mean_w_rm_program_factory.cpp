// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_mean_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::reduction::reduce_mean_w_rm::program {

using namespace tt;
using namespace tt::tt_metal;

// Factory implementation for Stage 5-6
ReduceMeanWRmProgramFactory::cached_program_t ReduceMeanWRmProgramFactory::create(
    const ReduceMeanWRmParams& operation_attributes,
    const ReduceMeanWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
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
    const uint32_t Wt = W / tt::constants::TILE_WIDTH;
    // Note: H and Ht will be used in Stage 6 kernel args

    // Stick sizes (unaligned and aligned for NoC)
    const uint32_t input_stick_size = W * input.element_size();
    const uint32_t input_stick_size_aligned = tt::round_up(input_stick_size, src_buffer->alignment());
    const uint32_t output_stick_size = 32 * output.element_size();  // Output width padded to 32
    const uint32_t output_stick_size_aligned = tt::round_up(output_stick_size, dst_buffer->alignment());

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
    // 4. Create circular buffers
    //    - CB c_0: Input row-major sticks (buffering_factor * TILE_HEIGHT = 64 sticks per tile-row)
    //    - CB c_1: Tiled input (buffering_factor * Wt tiles)
    //    - CB c_2: Scaler tile (1 tile, persistent)
    //    - CB c_3: Reduced tiled output (1 tile, width after reduction)
    //    - CB c_16: Output row-major sticks (1 tile, width = 32 padded)
    // ============================================================

    // CB c_0: Input RM sticks (double-buffered, 32 sticks per tile row)
    constexpr uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
    const uint32_t cb_in_rm_page_size = input_stick_size_aligned;
    const uint32_t cb_in_rm_num_pages = buffering_factor * tt::constants::TILE_HEIGHT;  // 2 * 32 = 64
    tt::tt_metal::create_cb(cb_in_rm_idx, program, all_cores, cb_in_rm_page_size, cb_in_rm_num_pages, cb_data_format);

    // CB c_1: Tiled input (double-buffered, Wt tiles per tile-row)
    constexpr uint32_t cb_in_tiled_idx = tt::CBIndex::c_1;
    const uint32_t cb_in_tiled_page_size = tile_size;
    const uint32_t cb_in_tiled_num_pages = buffering_factor * Wt;
    tt::tt_metal::create_cb(
        cb_in_tiled_idx, program, all_cores, cb_in_tiled_page_size, cb_in_tiled_num_pages, cb_data_format);

    // CB c_2: Scaler tile (1 tile, persistent, contains 1/W for mean)
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    const uint32_t cb_scaler_page_size = tile_size;
    const uint32_t cb_scaler_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_scaler_idx, program, all_cores, cb_scaler_page_size, cb_scaler_num_pages, cb_data_format);

    // CB c_3: Reduced tiled output (1 tile, output width = 1 tile after reduction)
    constexpr uint32_t cb_reduced_tiled_idx = tt::CBIndex::c_3;
    const uint32_t cb_reduced_tiled_page_size = tile_size;
    const uint32_t cb_reduced_tiled_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_reduced_tiled_idx,
        program,
        all_cores,
        cb_reduced_tiled_page_size,
        cb_reduced_tiled_num_pages,
        cb_data_format);

    // CB c_16: Output RM sticks (1 tile, output width = 32 padded)
    constexpr uint32_t cb_out_rm_idx = tt::CBIndex::c_16;
    const uint32_t cb_out_rm_page_size = output_stick_size_aligned;
    const uint32_t cb_out_rm_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_out_rm_idx, program, all_cores, cb_out_rm_page_size, cb_out_rm_num_pages, cb_data_format);

    // ============================================================
    // 5. Create kernels (Stage 6 - will throw here until kernels exist)
    // ============================================================
    TT_THROW("reduce_mean_w_rm: Kernel creation not yet implemented (Stage 6)");

    // The code below will be uncommented in Stage 6
    /*
    // Compile-time args for kernels
    std::vector<uint32_t> reader_compile_time_args = {...};
    std::vector<uint32_t> compute_compile_time_args = {...};
    std::vector<uint32_t> writer_compile_time_args = {...};

    // Create kernels
    const auto reader_id = tt::tt_metal::CreateKernel(...);
    const auto compute_id = tt::tt_metal::CreateKernel(...);
    const auto writer_id = tt::tt_metal::CreateKernel(...);

    // Set runtime args
    ...

    return {
        std::move(program),
        ReduceMeanWRmSharedVariables{
            .reader_kernel_id = reader_id,
            .compute_kernel_id = compute_id,
            .writer_kernel_id = writer_id,
            .all_cores = all_cores,
            .num_cores = 1}};
    */
}

void ReduceMeanWRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ReduceMeanWRmParams& operation_attributes,
    const ReduceMeanWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    (void)cached_program;
    (void)operation_attributes;
    (void)tensor_args;
    (void)tensor_return_value;
    // Stub - update runtime arguments for cached program
    // This will update tensor buffer addresses when program is reused
}

}  // namespace ttnn::operations::reduction::reduce_mean_w_rm::program
