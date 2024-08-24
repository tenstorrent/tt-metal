// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/cpp/ttnn/operations/core/work_split/work_split.hpp"

namespace ttnn::operations::experimental::ssm::detail {

using namespace tt::constants;

operation::ProgramWithCallbacks multi_core_ssm_1d_sum_reduce(
    const Tensor& a, Tensor& output, MathFidelity math_fidelity, CoreCoord compute_with_storage_grid_size) {
    constexpr uint32_t ONE_TILE = 1;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t LATENT_DIM = TILE_WIDTH;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto* input_buffer = a.buffer();
    const bool input_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    tt::tt_metal::Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");
    const bool output_is_dram = out_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    auto ashape = a.get_legacy_shape();
    auto num_output_blocks_total = a.get_legacy_shape()[-1] / (TILE_WIDTH * TILE_WIDTH);

    const bool row_major = false;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
            ttnn::split_work_to_cores(compute_with_storage_grid_size, num_output_blocks_total, row_major);

    const auto create_circular_buffer = [&program, &cores = all_cores](
                                            uint32_t index,
                                            uint32_t num_tiles,
                                            uint32_t tile_size,
                                            const tt::DataFormat& format) -> tt::tt_metal::CBHandle {
        const tt::tt_metal::CircularBufferConfig config =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{index, format}})
                .set_page_size(index, tile_size);
        return tt::tt_metal::CreateCircularBuffer(program, cores, config);
    };

    TT_ASSERT(a.get_dtype() == output.get_dtype(), "Input and output tensors must be of same type");

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    const uint32_t input_tile_size = tt::tt_metal::detail::TileSize(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tt_metal::detail::TileSize(intermediary_format);

    const uint32_t cb_size = 2;

    // Reader writes input tiles to this
    const uint32_t input_cb_id = tt::CB::c_in0;
    const auto input_cb = create_circular_buffer(input_cb_id, cb_size, input_tile_size, input_format);

    // Reader writes scaling tile to this CB. We need it because the reduce LLK requires a scaling factor tile.
    const uint32_t scalar_cb_id = tt::CB::c_in2;
    const auto scalar_cb = create_circular_buffer(scalar_cb_id, cb_size, intermediary_tile_size, intermediary_format);

    // Compute writes transposed tile (loopback)
    const uint32_t intermed_cb_id0 = tt::CB::c_intermed0;
    const auto intermed_cb0 =
        create_circular_buffer(intermed_cb_id0, cb_size, intermediary_tile_size, intermediary_format);

    // Compute writes reduced tile for writer
    const uint32_t intermed_cb_id1 = tt::CB::c_intermed1;
    const auto intermed_cb1 =
        create_circular_buffer(intermed_cb_id1, cb_size, intermediary_tile_size, intermediary_format);

    // Writer concats and writes back to compute
    const uint32_t intermed_cb_id2 = tt::CB::c_intermed2;
    const auto intermed_cb2 =
        create_circular_buffer(intermed_cb_id2, cb_size, intermediary_tile_size, intermediary_format);

    // Compute transposes and writes back to writer
    const uint32_t output_cb_id = tt::CB::c_out0;
    const auto output_cb = create_circular_buffer(output_cb_id, cb_size, input_tile_size, input_format);

    const bfloat16 bfloat_scaler_value = bfloat16(1.0f);
    const uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    std::vector<uint32_t> reader_compile_time_args = {input_is_dram, packed_scaler_value};
    std::vector<uint32_t> writer_compile_time_args = {
        intermed_cb_id1,
        intermed_cb_id2,
        output_cb_id,
        output_is_dram,
    };
    std::vector<uint32_t> compute_compile_time_args = {
        input_cb_id,
        scalar_cb_id,
        intermed_cb_id0,
        intermed_cb_id1,
        intermed_cb_id2,
        output_cb_id,
    };

    // Reuse the reader from reduce since we want the same behavior
    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/reader_ssm_1d_sum_reduce.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/writer_ssm_1d_sum_reduce.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/ssm_1d_sum_reduce.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();
    std::vector<CoreCoord> cores =
        grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, row_major);
    auto set_runtime_args = [reader_kernel_id,
                             writer_kernel_id,
                             compute_kernel_id,
                             num_cores = num_cores,
                             all_cores = all_cores,
                             cores = cores,
                             g1_numcores = g1_numcores,
                             g2_numcores = g2_numcores,
                             num_blocks_per_core_group_1 = num_blocks_per_core_group_1,
                             num_blocks_per_core_group_2 = num_blocks_per_core_group_2,
                             ashape = ashape](Program& program, const Tensor& a, const Tensor& output) {
        tt::tt_metal::Buffer* input_buffer = a.buffer();
        tt::tt_metal::Buffer* output_buffer = output.buffer();

        uint32_t num_blocks_per_core = 0;

        std::vector<std::vector<uint32_t>> reader_runtime_args = {
            cores.size(), {0, 0, 0, 0, 0}};  // (src_addr, num_tiles, start_id)

        std::vector<std::vector<uint32_t>> writer_runtime_args = {
            cores.size(), {0, 0, 0, 0, 0}};  // (dst_addr, num_tiles, start_id)

        std::vector<std::vector<uint32_t>> compute_runtime_args = {cores.size(), {0, 0}};

        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
            const CoreCoord& core = cores.at(i);

            if (i < g1_numcores) {
                num_blocks_per_core = num_blocks_per_core_group_1;
            } else {
                num_blocks_per_core = num_blocks_per_core_group_2;
            }

            reader_runtime_args[i][0] = input_buffer->address();
            reader_runtime_args[i][1] = num_blocks_per_core * LATENT_DIM;
            reader_runtime_args[i][2] = num_blocks_written * LATENT_DIM;
            reader_runtime_args[i][3] = ashape[2] / TILE_HEIGHT;
            reader_runtime_args[i][4] = ashape[-1] / TILE_WIDTH;

            writer_runtime_args[i][0] = output_buffer->address();
            writer_runtime_args[i][1] = num_blocks_per_core;
            writer_runtime_args[i][2] = num_blocks_written;
            writer_runtime_args[i][3] = ashape[2] / TILE_HEIGHT;
            writer_runtime_args[i][4] = ashape[-1] / (LATENT_DIM * TILE_WIDTH);

            compute_runtime_args[i][0] = num_blocks_per_core;
            compute_runtime_args[i][1] = ashape[2] / TILE_HEIGHT;

            num_blocks_written += num_blocks_per_core;
        }

        SetRuntimeArgs(program, reader_kernel_id, cores, reader_runtime_args);
        SetRuntimeArgs(program, writer_kernel_id, cores, writer_runtime_args);
        SetRuntimeArgs(program, compute_kernel_id, cores, compute_runtime_args);
    };

    set_runtime_args(program, a, output);

    auto override_runtime_arguments_callback = [set_runtime_args](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& output_tensor = output_tensors.at(0);
        set_runtime_args(program, input_tensors.at(0), output_tensor);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::ssm::detail
