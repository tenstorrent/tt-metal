// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/madd/device/madd_program_factory_sharded.hpp"

namespace ttnn::prim {

MAddProgramFactorySharded::cached_program_t MAddProgramFactorySharded::create(
    const MAddParams& operation_attributes, const MAddArgs& tensor_args, Tensor& output_tensor) {
    const ttnn::Tensor& a = tensor_args.a;
    const ttnn::Tensor& b = tensor_args.b;
    const ttnn::Tensor& c = tensor_args.c;
    const ttnn::Tensor& output = output_tensor;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Get shard spec - all inputs guaranteed to have identical sharding by validation
    const auto shard_spec = a.shard_spec().value();
    const auto all_cores = shard_spec.grid;

    // Calculate tiles per core from shard shape
    const uint32_t shard_height = shard_spec.shape[0];
    const uint32_t shard_width = shard_spec.shape[1];
    const uint32_t num_tiles_per_core = (shard_height * shard_width) / tt::constants::TILE_HW;

    // Data formats
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    const uint32_t input_tile_size = tt::tile_size(input_cb_data_format);
    const uint32_t output_tile_size = tt::tile_size(output_cb_data_format);

    // Create sharded circular buffers - data already in L1
    uint32_t next_cb_index = tt::CBIndex::c_0;

    // Input A CB - sharded
    tt::tt_metal::CircularBufferConfig cb_srcA_config =
        tt::tt_metal::CircularBufferConfig(
            num_tiles_per_core * input_tile_size, {{next_cb_index, input_cb_data_format}})
            .set_page_size(next_cb_index, input_tile_size)
            .set_globally_allocated_address(*a.buffer());
    const auto cb_srcA = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_srcA_config);
    const uint32_t cb_srcA_index = next_cb_index++;

    // Input B CB - sharded
    tt::tt_metal::CircularBufferConfig cb_srcB_config =
        tt::tt_metal::CircularBufferConfig(
            num_tiles_per_core * input_tile_size, {{next_cb_index, input_cb_data_format}})
            .set_page_size(next_cb_index, input_tile_size)
            .set_globally_allocated_address(*b.buffer());
    const auto cb_srcB = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_srcB_config);
    const uint32_t cb_srcB_index = next_cb_index++;

    // Input C CB - sharded
    tt::tt_metal::CircularBufferConfig cb_srcC_config =
        tt::tt_metal::CircularBufferConfig(
            num_tiles_per_core * input_tile_size, {{next_cb_index, input_cb_data_format}})
            .set_page_size(next_cb_index, input_tile_size)
            .set_globally_allocated_address(*c.buffer());
    const auto cb_srcC = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_srcC_config);
    const uint32_t cb_srcC_index = next_cb_index++;

    // Zero tile CB (not sharded, local scratch space)
    const auto [cb_zero_index, cb_zero] =
        tt::tt_metal::create_cb(next_cb_index++, program, all_cores, input_tile_size, 1, input_cb_data_format);

    // Output CB - sharded
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_tiles_per_core * output_tile_size, {{next_cb_index, output_cb_data_format}})
            .set_page_size(next_cb_index, output_tile_size)
            .set_globally_allocated_address(*output.buffer());
    const auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    const uint32_t cb_output_index = next_cb_index++;

    // Create sharded reader kernel - data already in L1, just signal availability
    const std::vector<uint32_t> reader_compile_time_args = {cb_srcA_index, cb_srcB_index, cb_srcC_index, cb_zero_index};

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/madd/device/kernels/dataflow/reader_madd_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create compute kernel - same as interleaved, processes all tiles on this core
    const std::vector<uint32_t> compute_compile_time_args = {
        num_tiles_per_core, cb_srcA_index, cb_srcB_index, cb_srcC_index, cb_zero_index, cb_output_index};

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/madd/device/kernels/compute/madd_compute_sharded.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args});

    // Create sharded writer kernel - data stays in L1, just wait for compute to finish
    const std::vector<uint32_t> writer_compile_time_args = {cb_output_index};

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/madd/device/kernels/dataflow/writer_madd_sharded.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime arguments - just num_tiles_per_core for reader and writer
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt, true)) {
        tt::tt_metal::SetRuntimeArgs(program, 0, core, {num_tiles_per_core});  // reader
        tt::tt_metal::SetRuntimeArgs(program, 2, core, {num_tiles_per_core});  // writer
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{.cb_srcA = cb_srcA, .cb_srcB = cb_srcB, .cb_srcC = cb_srcC, .cb_output = cb_output}};
}

void MAddProgramFactorySharded::override_runtime_arguments(
    cached_program_t& cached_program,
    [[maybe_unused]] const MAddParams& operation_attributes,
    const MAddArgs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    // Update CB addresses for new tensor allocations
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, shared_vars.cb_srcA, *tensor_args.a.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, shared_vars.cb_srcB, *tensor_args.b.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, shared_vars.cb_srcC, *tensor_args.c.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, shared_vars.cb_output, *output_tensor.buffer());
}

}  // namespace ttnn::prim
