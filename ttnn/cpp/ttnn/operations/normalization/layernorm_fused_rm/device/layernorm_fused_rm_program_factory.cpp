// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fused_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::normalization::layernorm_fused_rm::program {

using namespace tt;
using namespace tt::tt_metal;

LayernormFusedRmProgramFactory::cached_program_t LayernormFusedRmProgramFactory::create(
    const LayernormFusedRmParams& operation_attributes,
    const LayernormFusedRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    auto& output = tensor_return_value;

    auto src_buffer = input.buffer();
    auto gamma_buffer = gamma.buffer();
    auto beta_buffer = beta.buffer();
    auto dst_buffer = output.buffer();

    tt::tt_metal::Program program{};

    // Data formats
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t tile_size = tt::tile_size(cb_data_format);

    // Shape extraction
    const auto& input_shape = input.logical_shape();
    uint32_t W = input_shape[-1];                  // Width (last dimension)
    uint32_t H = input_shape[-2];                  // Height (second-to-last dimension)
    uint32_t Wt = W / tt::constants::TILE_WIDTH;   // Tiles per row
    uint32_t Ht = H / tt::constants::TILE_HEIGHT;  // Tile rows

    // Work distribution - single core for initial implementation
    CoreRangeSet all_cores = CoreRangeSet({CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})});
    uint32_t num_cores = 1;

    // Stick size for row-major data
    uint32_t stick_size = W * input.element_size();  // Size of one RM stick in bytes

    // Circular Buffer Configuration (from spec)
    // CB c_0: Input RM sticks (32 sticks per tile row, each stick is W elements)
    // Reader pushes input TWICE per tile row (for mean+center+square+var, then for re-center+normalize)
    // Double-buffered: 2 * 32 sticks
    uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
    uint32_t cb_in_rm_size = 2 * 32 * stick_size;  // 2x 32 sticks per tile row
    tt::tt_metal::CircularBufferConfig cb_in_rm_config =
        tt::tt_metal::CircularBufferConfig(cb_in_rm_size, {{cb_in_rm_idx, cb_data_format}})
            .set_page_size(cb_in_rm_idx, stick_size);  // Page = one stick
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_rm_config);

    // CB c_1: Tiled input (double-buffered, 2*Wt tiles)
    uint32_t cb_in_tiled_idx = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_in_tiled_config =
        tt::tt_metal::CircularBufferConfig(2 * Wt * tile_size, {{cb_in_tiled_idx, cb_data_format}})
            .set_page_size(cb_in_tiled_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_tiled_config);

    // CB c_2: Scaler (1/W) - 2 tiles for double buffering
    uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_scaler_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{cb_scaler_idx, cb_data_format}})
            .set_page_size(cb_scaler_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    // CB c_3: Epsilon tile - 1 tile
    uint32_t cb_eps_idx = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_eps_config =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_eps_idx, cb_data_format}})
            .set_page_size(cb_eps_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_eps_config);

    // CB c_4: Gamma RM sticks (one stick, gamma is 1D with width W)
    uint32_t cb_gamma_rm_idx = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_gamma_rm_config =
        tt::tt_metal::CircularBufferConfig(stick_size, {{cb_gamma_rm_idx, cb_data_format}})
            .set_page_size(cb_gamma_rm_idx, stick_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_gamma_rm_config);

    // CB c_5: Beta RM sticks (one stick, beta is 1D with width W)
    uint32_t cb_beta_rm_idx = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_beta_rm_config =
        tt::tt_metal::CircularBufferConfig(stick_size, {{cb_beta_rm_idx, cb_data_format}})
            .set_page_size(cb_beta_rm_idx, stick_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_beta_rm_config);

    // CB c_6: Tiled gamma (persistent, Wt tiles)
    uint32_t cb_gamma_tiled_idx = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig cb_gamma_tiled_config =
        tt::tt_metal::CircularBufferConfig(Wt * tile_size, {{cb_gamma_tiled_idx, cb_data_format}})
            .set_page_size(cb_gamma_tiled_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_gamma_tiled_config);

    // CB c_7: Tiled beta (persistent, Wt tiles)
    uint32_t cb_beta_tiled_idx = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig cb_beta_tiled_config =
        tt::tt_metal::CircularBufferConfig(Wt * tile_size, {{cb_beta_tiled_idx, cb_data_format}})
            .set_page_size(cb_beta_tiled_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_beta_tiled_config);

    // CB c_16: Output RM sticks (Wt tile-sized pages to match untilize helper)
    // Total capacity: Wt * tile_size = 32 * stick_size (same as before)
    uint32_t cb_out_rm_idx = tt::CBIndex::c_16;
    uint32_t cb_out_rm_size = Wt * tile_size;  // Wt tile-sized pages
    tt::tt_metal::CircularBufferConfig cb_out_rm_config =
        tt::tt_metal::CircularBufferConfig(cb_out_rm_size, {{cb_out_rm_idx, cb_data_format}})
            .set_page_size(cb_out_rm_idx, tile_size);  // Page = tile_size (for untilize helper compatibility)
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_rm_config);

    // Intermediate CBs for layernorm compute
    // CB c_24: Centered (x - mean), Wt tiles
    uint32_t cb_centered_idx = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig cb_centered_config =
        tt::tt_metal::CircularBufferConfig(Wt * tile_size, {{cb_centered_idx, cb_data_format}})
            .set_page_size(cb_centered_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_centered_config);

    // CB c_25: Mean tile, 1 tile
    uint32_t cb_mean_idx = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig cb_mean_config =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_mean_idx, cb_data_format}})
            .set_page_size(cb_mean_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_mean_config);

    // CB c_26: Variance tile, 1 tile
    uint32_t cb_var_idx = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig cb_var_config =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_var_idx, cb_data_format}})
            .set_page_size(cb_var_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_var_config);

    // CB c_27: Inverse std (1/sqrt(var+eps)), 1 tile
    uint32_t cb_invstd_idx = tt::CBIndex::c_27;
    tt::tt_metal::CircularBufferConfig cb_invstd_config =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_invstd_idx, cb_data_format}})
            .set_page_size(cb_invstd_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_invstd_config);

    // Compile-time arguments for reader kernel
    std::vector<uint32_t> reader_compile_time_args = {stick_size, Wt};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*gamma_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*beta_buffer).append_to(reader_compile_time_args);

    // Compile-time arguments for compute kernel
    std::vector<uint32_t> compute_compile_time_args = {Wt, Ht, W};

    // Compile-time arguments for writer kernel
    std::vector<uint32_t> writer_compile_time_args = {cb_out_rm_idx, stick_size, tt::constants::TILE_HEIGHT, Wt};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Create kernels
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/device/kernels/dataflow/"
        "reader_layernorm_fused_rm.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/device/kernels/compute/layernorm_fused_rm.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/device/kernels/dataflow/"
        "writer_layernorm_fused_rm.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Pack epsilon as bfloat16
    union {
        float f;
        uint32_t u;
    } epsilon_union;
    epsilon_union.f = operation_attributes.epsilon;
    uint32_t epsilon_packed = epsilon_union.u;

    // Pack scaler (1/W) as bfloat16
    union {
        float f;
        uint32_t u;
    } scaler_union;
    scaler_union.f = 1.0f / static_cast<float>(W);
    uint32_t scaler_packed = scaler_union.u;

    // Runtime arguments for reader kernel
    uint32_t num_tile_rows = Ht;
    uint32_t start_stick_id = 0;
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

    // Runtime arguments for compute kernel
    tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, CoreCoord{0, 0}, {num_tile_rows});

    // Runtime arguments for writer kernel
    tt::tt_metal::SetRuntimeArgs(
        program, writer_kernel_id, CoreCoord{0, 0}, {dst_buffer->address(), num_tile_rows, start_stick_id});

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

    auto src_buffer = input.buffer();
    auto gamma_buffer = gamma.buffer();
    auto beta_buffer = beta.buffer();
    auto dst_buffer = output.buffer();

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
