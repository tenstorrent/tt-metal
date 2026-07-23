// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/kda_recurrent/device/kda_recurrent_program_factory.hpp"

#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/math.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

KDARecurrentProgramFactory::cached_program_t KDARecurrentProgramFactory::create(
    const KDARecurrentParams& attributes, const KDARecurrentInputs& inputs, std::vector<Tensor>& outputs) {
    Program program{};
    const uint32_t heads = attributes.num_heads;
    const uint32_t key_tiles = attributes.key_dim / TILE_WIDTH;
    const uint32_t value_tiles = attributes.value_dim / TILE_WIDTH;
    const uint32_t state_tiles = key_tiles * value_tiles;

    auto* device = inputs.q_scaled.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    TT_FATAL(heads <= grid.x * grid.y, "num_heads {} exceeds available compute cores {}", heads, grid.x * grid.y);

    std::vector<CoreCoord> head_cores(heads);
    std::set<CoreRange> core_ranges;
    for (uint32_t head = 0; head < heads; ++head) {
        head_cores[head] = CoreCoord{head / grid.y, head % grid.y};
        core_ranges.insert(CoreRange{head_cores[head], head_cores[head]});
    }
    const CoreRangeSet cores{core_ranges};
    constexpr auto data_format = tt::DataFormat::Float32;

    auto create_cb = [&](uint32_t index, uint32_t tiles, uint32_t buffers = 1) {
        const uint32_t tile_size = tt::tile_size(data_format);
        CircularBufferConfig config(tiles * buffers * tile_size, {{index, data_format}});
        config.set_page_size(index, tile_size);
        CreateCircularBuffer(program, cores, config);
    };

    create_cb(0, key_tiles);     // q_scaled
    create_cb(1, key_tiles);     // k_unit
    create_cb(2, value_tiles);   // v
    create_cb(3, key_tiles);     // decay column
    create_cb(4, 1);             // beta scalar
    create_cb(5, state_tiles);   // input state
    create_cb(6, state_tiles);   // decayed state
    create_cb(7, value_tiles);   // v_read
    create_cb(8, value_tiles);   // delta
    create_cb(9, value_tiles);   // beta * delta
    create_cb(10, key_tiles);    // transposed k
    create_cb(11, state_tiles);  // rank-one update
    create_cb(12, state_tiles);  // updated state, compute-local
    create_cb(13, value_tiles);  // output
    create_cb(14, state_tiles);  // final state, writer-owned

    const std::string kernel_directory = "ttnn/cpp/ttnn/operations/transformer/kda_recurrent/device/kernels/";
    const std::vector<uint32_t> compute_time_args = {key_tiles, value_tiles};

    std::vector<uint32_t> reader_compile_args = compute_time_args;
    TensorAccessorArgs(inputs.q_scaled.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(inputs.k_unit.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(inputs.v.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(inputs.decay.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(inputs.beta.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(inputs.state.buffer()).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = compute_time_args;
    TensorAccessorArgs(outputs[0].buffer()).append_to(writer_compile_args);
    TensorAccessorArgs(outputs[1].buffer()).append_to(writer_compile_args);

    const auto reader_kernel = CreateKernel(
        program,
        kernel_directory + "dataflow/reader_kda_recurrent.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});
    const auto writer_kernel = CreateKernel(
        program,
        kernel_directory + "dataflow/writer_kda_recurrent.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);
    const auto compute_kernel = CreateKernel(
        program,
        kernel_directory + "compute/kda_recurrent.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_time_args});

    const uint32_t q_address = inputs.q_scaled.buffer()->address();
    const uint32_t k_address = inputs.k_unit.buffer()->address();
    const uint32_t v_address = inputs.v.buffer()->address();
    const uint32_t decay_address = inputs.decay.buffer()->address();
    const uint32_t beta_address = inputs.beta.buffer()->address();
    const uint32_t state_address = inputs.state.buffer()->address();
    const uint32_t output_address = outputs[0].buffer()->address();
    const uint32_t final_state_address = outputs[1].buffer()->address();

    for (uint32_t head = 0; head < heads; ++head) {
        const auto& core = head_cores[head];
        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {head, q_address, k_address, v_address, decay_address, beta_address, state_address});
        SetRuntimeArgs(program, writer_kernel, core, {head, output_address, final_state_address});
        SetRuntimeArgs(program, compute_kernel, core, {});
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernel_id = reader_kernel,
            .writer_kernel_id = writer_kernel,
            .grid_y = grid.y,
        }};
}

void KDARecurrentProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const KDARecurrentParams& attributes,
    const KDARecurrentInputs& inputs,
    std::vector<Tensor>& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const uint32_t q_address = inputs.q_scaled.buffer()->address();
    const uint32_t k_address = inputs.k_unit.buffer()->address();
    const uint32_t v_address = inputs.v.buffer()->address();
    const uint32_t decay_address = inputs.decay.buffer()->address();
    const uint32_t beta_address = inputs.beta.buffer()->address();
    const uint32_t state_address = inputs.state.buffer()->address();
    const uint32_t output_address = outputs[0].buffer()->address();
    const uint32_t final_state_address = outputs[1].buffer()->address();

    for (uint32_t head = 0; head < attributes.num_heads; ++head) {
        const CoreCoord core{head / shared.grid_y, head % shared.grid_y};
        auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id, core);
        reader_args[1] = q_address;
        reader_args[2] = k_address;
        reader_args[3] = v_address;
        reader_args[4] = decay_address;
        reader_args[5] = beta_address;
        reader_args[6] = state_address;

        auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id, core);
        writer_args[1] = output_address;
        writer_args[2] = final_state_address;
    }
}

}  // namespace ttnn::prim
