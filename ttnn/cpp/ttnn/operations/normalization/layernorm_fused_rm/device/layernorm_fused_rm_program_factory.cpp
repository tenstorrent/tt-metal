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

    // Stick size for row-major data
    const uint32_t stick_size = W * input.element_size();  // Size of one RM stick in bytes

    // ============================================================
    // 2. Create program
    // ============================================================
    tt::tt_metal::Program program{};

    // ============================================================
    // 3. Circular Buffer Configuration (modern API)
    // ============================================================

    // CB c_0: Input RM sticks (32 sticks per tile row, each stick is W elements)
    // Reader pushes input TWICE per tile row (for mean+center+square+var, then for re-center+normalize)
    // Double-buffered: 2 * 32 sticks
    constexpr uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
    const uint32_t cb_in_rm_pages = buffering_factor * tt::constants::TILE_HEIGHT;
    tt::tt_metal::create_cb(cb_in_rm_idx, program, all_cores, stick_size, cb_in_rm_pages, cb_data_format);

    // CB c_1: Tiled input (double-buffered, 2*Wt tiles)
    constexpr uint32_t cb_in_tiled_idx = tt::CBIndex::c_1;
    tt::tt_metal::create_cb(cb_in_tiled_idx, program, all_cores, tile_size, buffering_factor * Wt, cb_data_format);

    // CB c_2: Scaler (1/W) - 1 tile (read once, used many times)
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    tt::tt_metal::create_cb(cb_scaler_idx, program, all_cores, tile_size, 1, cb_data_format);

    // CB c_3: Epsilon tile - 1 tile
    constexpr uint32_t cb_eps_idx = tt::CBIndex::c_3;
    tt::tt_metal::create_cb(cb_eps_idx, program, all_cores, tile_size, 1, cb_data_format);

    // CB c_4: Gamma RM sticks (one stick, gamma is 1D with width W)
    constexpr uint32_t cb_gamma_rm_idx = tt::CBIndex::c_4;
    tt::tt_metal::create_cb(cb_gamma_rm_idx, program, all_cores, stick_size, 1, cb_data_format);

    // CB c_5: Beta RM sticks (one stick, beta is 1D with width W)
    constexpr uint32_t cb_beta_rm_idx = tt::CBIndex::c_5;
    tt::tt_metal::create_cb(cb_beta_rm_idx, program, all_cores, stick_size, 1, cb_data_format);

    // CB c_6: Tiled gamma (persistent, Wt tiles)
    constexpr uint32_t cb_gamma_tiled_idx = tt::CBIndex::c_6;
    tt::tt_metal::create_cb(cb_gamma_tiled_idx, program, all_cores, tile_size, Wt, cb_data_format);

    // CB c_7: Tiled beta (persistent, Wt tiles)
    constexpr uint32_t cb_beta_tiled_idx = tt::CBIndex::c_7;
    tt::tt_metal::create_cb(cb_beta_tiled_idx, program, all_cores, tile_size, Wt, cb_data_format);

    // CB c_16: Output RM sticks (Wt tile-sized pages to match untilize helper)
    constexpr uint32_t cb_out_rm_idx = tt::CBIndex::c_16;
    tt::tt_metal::create_cb(cb_out_rm_idx, program, all_cores, tile_size, Wt, cb_data_format);

    // Intermediate CBs for layernorm compute
    // CB c_24: Centered (x - mean), Wt tiles
    constexpr uint32_t cb_centered_idx = tt::CBIndex::c_24;
    tt::tt_metal::create_cb(cb_centered_idx, program, all_cores, tile_size, Wt, cb_data_format);

    // CB c_25: Mean tile, 1 tile
    constexpr uint32_t cb_mean_idx = tt::CBIndex::c_25;
    tt::tt_metal::create_cb(cb_mean_idx, program, all_cores, tile_size, 1, cb_data_format);

    // CB c_26: Variance tile, 1 tile
    constexpr uint32_t cb_var_idx = tt::CBIndex::c_26;
    tt::tt_metal::create_cb(cb_var_idx, program, all_cores, tile_size, 1, cb_data_format);

    // CB c_27: Inverse std (1/sqrt(var+eps)), 1 tile
    constexpr uint32_t cb_invstd_idx = tt::CBIndex::c_27;
    tt::tt_metal::create_cb(cb_invstd_idx, program, all_cores, tile_size, 1, cb_data_format);

    // ============================================================
    // 4. Create kernels
    // ============================================================

    // Compile-time arguments for reader kernel
    std::vector<uint32_t> reader_compile_time_args = {stick_size, Wt};
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
    std::vector<uint32_t> writer_compile_time_args = {cb_out_rm_idx, stick_size, tt::constants::TILE_HEIGHT, Wt};
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
