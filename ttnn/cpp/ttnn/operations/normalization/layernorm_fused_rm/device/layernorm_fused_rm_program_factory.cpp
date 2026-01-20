// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fused_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::normalization::layernorm_fused_rm::program {

using namespace tt;
using namespace tt::tt_metal;

LayernormFusedRmProgramFactory::cached_program_t LayernormFusedRmProgramFactory::create(
    const LayernormFusedRmParams& operation_attributes,
    const LayernormFusedRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    // ============================================================
    // CONSTANTS
    // ============================================================
    constexpr uint32_t buffering_factor = 2;  // Double buffering

    // ============================================================
    // 1. Extract tensor properties (ALL const)
    // ============================================================
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    auto& output = tensor_return_value;

    const auto* src_buffer = input.buffer();
    const auto* gamma_buffer = gamma.buffer();
    const auto* beta_buffer = beta.buffer();
    const auto* dst_buffer = output.buffer();

    // Data formats
    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(cb_data_format);

    // Shape extraction
    const auto& input_shape = input.logical_shape();
    const uint32_t W = input_shape[-1];                  // Width (last dimension)
    const uint32_t H = input_shape[-2];                  // Height (second-to-last dimension)
    const uint32_t Wt = W / tt::constants::TILE_WIDTH;   // Tiles per row
    const uint32_t Ht = H / tt::constants::TILE_HEIGHT;  // Tile rows

    // Work distribution - single core for initial implementation
    const CoreRangeSet all_cores = CoreRangeSet({CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})});
    const uint32_t num_cores = 1;

    // Stick sizes for row-major data (aligned for NoC efficiency)
    const uint32_t input_stick_size = W * input.element_size();
    const uint32_t input_stick_size_aligned = tt::round_up(input_stick_size, src_buffer->alignment());

    // ============================================================
    // 2. Create program
    // ============================================================
    tt::tt_metal::Program program{};

    // ============================================================
    // 3. Circular Buffer Configuration (modern API)
    // ============================================================

    // CB c_0: Input RM sticks - Reader pushes input TWICE per tile row
    // (for mean+center+square+var, then for re-center+normalize)
    constexpr uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
    const uint32_t cb_in_rm_page_size = input_stick_size_aligned;
    const uint32_t cb_in_rm_num_pages = buffering_factor * tt::constants::TILE_HEIGHT;  // 2 * 32 sticks
    tt::tt_metal::create_cb(cb_in_rm_idx, program, all_cores, cb_in_rm_page_size, cb_in_rm_num_pages, cb_data_format);

    // CB c_1: Tiled input (double-buffered)
    constexpr uint32_t cb_in_tiled_idx = tt::CBIndex::c_1;
    const uint32_t cb_in_tiled_page_size = tile_size;
    const uint32_t cb_in_tiled_num_pages = buffering_factor * Wt;
    tt::tt_metal::create_cb(
        cb_in_tiled_idx, program, all_cores, cb_in_tiled_page_size, cb_in_tiled_num_pages, cb_data_format);

    // CB c_2: Scaler (1/W) - read once, used many times
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    const uint32_t cb_scaler_page_size = tile_size;
    const uint32_t cb_scaler_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_scaler_idx, program, all_cores, cb_scaler_page_size, cb_scaler_num_pages, cb_data_format);

    // CB c_3: Epsilon tile
    constexpr uint32_t cb_eps_idx = tt::CBIndex::c_3;
    const uint32_t cb_eps_page_size = tile_size;
    const uint32_t cb_eps_num_pages = 1;
    tt::tt_metal::create_cb(cb_eps_idx, program, all_cores, cb_eps_page_size, cb_eps_num_pages, cb_data_format);

    // CB c_4: Gamma RM sticks (1D with width W)
    constexpr uint32_t cb_gamma_rm_idx = tt::CBIndex::c_4;
    const uint32_t cb_gamma_rm_page_size = input_stick_size_aligned;
    const uint32_t cb_gamma_rm_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_gamma_rm_idx, program, all_cores, cb_gamma_rm_page_size, cb_gamma_rm_num_pages, cb_data_format);

    // CB c_5: Beta RM sticks (1D with width W)
    constexpr uint32_t cb_beta_rm_idx = tt::CBIndex::c_5;
    const uint32_t cb_beta_rm_page_size = input_stick_size_aligned;
    const uint32_t cb_beta_rm_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_beta_rm_idx, program, all_cores, cb_beta_rm_page_size, cb_beta_rm_num_pages, cb_data_format);

    // CB c_6: Tiled gamma (persistent, holds full row)
    constexpr uint32_t cb_gamma_tiled_idx = tt::CBIndex::c_6;
    const uint32_t cb_gamma_tiled_page_size = tile_size;
    const uint32_t cb_gamma_tiled_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_gamma_tiled_idx, program, all_cores, cb_gamma_tiled_page_size, cb_gamma_tiled_num_pages, cb_data_format);

    // CB c_7: Tiled beta (persistent, holds full row)
    constexpr uint32_t cb_beta_tiled_idx = tt::CBIndex::c_7;
    const uint32_t cb_beta_tiled_page_size = tile_size;
    const uint32_t cb_beta_tiled_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_beta_tiled_idx, program, all_cores, cb_beta_tiled_page_size, cb_beta_tiled_num_pages, cb_data_format);

    // CB c_16: Output RM sticks (tile-sized pages for untilize helper compatibility)
    constexpr uint32_t cb_out_rm_idx = tt::CBIndex::c_16;
    const uint32_t cb_out_rm_page_size = tile_size;
    const uint32_t cb_out_rm_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_out_rm_idx, program, all_cores, cb_out_rm_page_size, cb_out_rm_num_pages, cb_data_format);

    // Intermediate CBs for layernorm compute

    // CB c_24: Centered data (x - mean), holds full row
    constexpr uint32_t cb_centered_idx = tt::CBIndex::c_24;
    const uint32_t cb_centered_page_size = tile_size;
    const uint32_t cb_centered_num_pages = Wt;
    tt::tt_metal::create_cb(
        cb_centered_idx, program, all_cores, cb_centered_page_size, cb_centered_num_pages, cb_data_format);

    // CB c_25: Mean tile
    constexpr uint32_t cb_mean_idx = tt::CBIndex::c_25;
    const uint32_t cb_mean_page_size = tile_size;
    const uint32_t cb_mean_num_pages = 1;
    tt::tt_metal::create_cb(cb_mean_idx, program, all_cores, cb_mean_page_size, cb_mean_num_pages, cb_data_format);

    // CB c_26: Variance tile
    constexpr uint32_t cb_var_idx = tt::CBIndex::c_26;
    const uint32_t cb_var_page_size = tile_size;
    const uint32_t cb_var_num_pages = 1;
    tt::tt_metal::create_cb(cb_var_idx, program, all_cores, cb_var_page_size, cb_var_num_pages, cb_data_format);

    // CB c_27: Inverse std (1/sqrt(var+eps))
    constexpr uint32_t cb_invstd_idx = tt::CBIndex::c_27;
    const uint32_t cb_invstd_page_size = tile_size;
    const uint32_t cb_invstd_num_pages = 1;
    tt::tt_metal::create_cb(
        cb_invstd_idx, program, all_cores, cb_invstd_page_size, cb_invstd_num_pages, cb_data_format);

    // ============================================================
    // 4. Create kernels
    // ============================================================

    // Compile-time arguments for reader kernel
    std::vector<uint32_t> reader_compile_time_args = {input_stick_size_aligned, Wt};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*gamma_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*beta_buffer).append_to(reader_compile_time_args);

    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/device/kernels/dataflow/"
        "reader_layernorm_fused_rm.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Compile-time arguments for compute kernel
    const std::vector<uint32_t> compute_compile_time_args = {Wt, Ht, W};

    const auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/device/kernels/compute/layernorm_fused_rm.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    // Compile-time arguments for writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        cb_out_rm_idx, input_stick_size_aligned, tt::constants::TILE_HEIGHT, Wt};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/device/kernels/dataflow/"
        "writer_layernorm_fused_rm.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // ============================================================
    // 5. Set runtime arguments
    // ============================================================

    // Pack epsilon as float32 bits
    union {
        float f;
        uint32_t u;
    } epsilon_union;
    epsilon_union.f = operation_attributes.epsilon;
    const uint32_t epsilon_packed = epsilon_union.u;

    // Pack scaler (1/W) as float32 bits
    union {
        float f;
        uint32_t u;
    } scaler_union;
    scaler_union.f = 1.0f / static_cast<float>(W);
    const uint32_t scaler_packed = scaler_union.u;

    const uint32_t num_tile_rows = Ht;
    const uint32_t start_stick_id = 0;

    // Reader kernel runtime args
    tt::tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        CoreCoord{0, 0},
        {src_buffer->address(),
         gamma_buffer->address(),
         beta_buffer->address(),
         num_tile_rows,
         start_stick_id,
         scaler_packed,
         epsilon_packed});

    // Compute kernel runtime args
    tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, CoreCoord{0, 0}, {num_tile_rows});

    // Writer kernel runtime args
    tt::tt_metal::SetRuntimeArgs(
        program, writer_kernel_id, CoreCoord{0, 0}, {dst_buffer->address(), num_tile_rows, start_stick_id});

    // ============================================================
    // 6. Return cached program
    // ============================================================
    return {
        std::move(program),
        LayernormFusedRmSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .all_cores = all_cores,
            .num_cores = num_cores}};
}

void LayernormFusedRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayernormFusedRmParams& /*operation_attributes*/,
    const LayernormFusedRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    auto& output = tensor_return_value;

    const auto* src_buffer = input.buffer();
    const auto* gamma_buffer = gamma.buffer();
    const auto* beta_buffer = beta.buffer();
    const auto* dst_buffer = output.buffer();

    // Update reader kernel runtime args (addresses)
    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = src_buffer->address();
        runtime_args[1] = gamma_buffer->address();
        runtime_args[2] = beta_buffer->address();
    }

    // Update writer kernel runtime args (address)
    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::normalization::layernorm_fused_rm::program
