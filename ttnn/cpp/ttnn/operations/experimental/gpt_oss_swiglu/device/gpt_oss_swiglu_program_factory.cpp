// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt_oss_swiglu_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::gpt_oss_swiglu::program {

GptOssSwigluProgramFactory::cached_program_t GptOssSwigluProgramFactory::create(
    [[maybe_unused]] const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    const auto& gate = tensor_args.gate_tensor;
    const auto& up = tensor_args.up_tensor;
    auto& out = tensor_return_value;

    Program program = CreateProgram();

    // All three tensors share the same BLOCK_SHARDED layout (validated upstream).
    auto shard_spec = gate.shard_spec().value();
    auto core_range_set = shard_spec.grid;
    const uint32_t shard_h = shard_spec.shape[0];
    const uint32_t shard_w = shard_spec.shape[1];
    const uint32_t tiles_per_core_h = shard_h / tt::constants::TILE_HEIGHT;
    const uint32_t tiles_per_core_w = shard_w / tt::constants::TILE_WIDTH;
    const uint32_t tiles_per_core = tiles_per_core_h * tiles_per_core_w;

    auto data_format = datatype_to_dataformat_converter(gate.dtype());
    const uint32_t single_tile_size = tile_size(data_format);

    // Globally allocated CBs over the sharded buffers — compute kernel reads/writes
    // directly from L1 shards without any reader/writer dataflow kernels.
    constexpr uint32_t cb_gate_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_up_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_idx = tt::CBIndex::c_2;

    auto make_sharded_cb = [&](uint32_t cb_index, const Tensor& t) {
        CircularBufferConfig cfg = CircularBufferConfig(tiles_per_core * single_tile_size, {{cb_index, data_format}})
                                       .set_page_size(cb_index, single_tile_size)
                                       .set_globally_allocated_address(*t.buffer());
        return CreateCircularBuffer(program, core_range_set, cfg);
    };

    auto cb_gate = make_sharded_cb(cb_gate_idx, gate);
    auto cb_up = make_sharded_cb(cb_up_idx, up);
    auto cb_out = make_sharded_cb(cb_out_idx, out);

    // Sharded CBs are pre-populated at the L1 shard address, but the producer/consumer
    // CB protocol still needs to be balanced. The reader pushes the input CBs (gate,
    // up) at start; the writer waits and pops the output CB. No NOC traffic — these
    // run on NCRISC/BRISC purely as protocol stubs.
    std::vector<uint32_t> reader_ct_args{cb_gate_idx, cb_up_idx, tiles_per_core};
    auto reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/gpt_oss_swiglu/device/kernels/reader_swiglu.cpp",
        core_range_set,
        ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args{cb_out_idx, tiles_per_core};
    auto writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/gpt_oss_swiglu/device/kernels/writer_swiglu.cpp",
        core_range_set,
        WriterDataMovementConfig(writer_ct_args));

    std::vector<uint32_t> compute_ct_args{
        cb_gate_idx,
        cb_up_idx,
        cb_out_idx,
        tiles_per_core,
    };

    // Compute kernel runs on every core in the shard grid; each instance processes
    // tiles_per_core tile pairs (gate, up) -> out via the SwiGLU SFPU.
    auto compute_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/gpt_oss_swiglu/device/kernels/compute_swiglu.cpp",
        core_range_set,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
        });
    (void)reader_kernel;
    (void)writer_kernel;

    return {
        std::move(program),
        shared_variables_t{
            .compute_kernel = compute_kernel,
            .cb_gate = cb_gate,
            .cb_up = cb_up,
            .cb_out = cb_out,
        }};
}

void GptOssSwigluProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    [[maybe_unused]] const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;
    // Globally allocated CBs need their addresses refreshed when underlying tensor
    // buffers change (e.g. after deallocation/reallocation in a hot loop).
    UpdateDynamicCircularBufferAddress(program, shared.cb_gate, *tensor_args.gate_tensor.buffer());
    UpdateDynamicCircularBufferAddress(program, shared.cb_up, *tensor_args.up_tensor.buffer());
    UpdateDynamicCircularBufferAddress(program, shared.cb_out, *tensor_return_value.buffer());
}

}  // namespace ttnn::operations::experimental::gpt_oss_swiglu::program
