// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_program_factory.hpp"
#include "moe_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::moe::program {

MoEProgramFactory::cached_program_t MoEProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Get the cores for the program
    auto cores = tt::tt_metal::CoreRangeSet({tt::tt_metal::CoreRange({0, 0}, {7, 7})});

    // Get input tensor info for sharded CB configuration
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shard_spec = input_tensor.shard_spec().value();
    const auto& shard_shape = input_shard_spec.shape;

    // Calculate tiles per core for sharded input
    // shard_shape is [height, width] in elements
    uint32_t shard_height_tiles = shard_shape[0] / tt::constants::TILE_HEIGHT;
    uint32_t shard_width_tiles = shard_shape[1] / tt::constants::TILE_WIDTH;
    uint32_t input_tiles_per_core = shard_height_tiles * shard_width_tiles;

    // Get input data format and tile size
    tt::DataFormat input_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_tile_size = tt::tile_size(input_df);

    // Create CBs for the program
    // CBs used in the MOE operation
    /*
        ----------------------------------------------------------------------------------------
        |     Name      |   CB Index    |   Dtype    | Bytes/tile | Tiles/CB |  Total size (B) |
        ----------------------------------------------------------------------------------------
        | cb_r2c_w0     | CBIndex::c_0  | bfloat8_b  |    1024    |    2     |      2048       |
        | cb_s2c_in     | CBIndex::c_1  | bfloat8_b  |    1024    |    2     |      2048       |
        | cb_c2c_mm0    | CBIndex::c_2  | bfloat8_b  |    1024    |    1     |      1024       |
        | cb_c2c_mm1    | CBIndex::c_3  | bfloat8_b  |    1024    |    1     |      1024       |
        | cb_c2w_elt    | CBIndex::c_4  | bfloat8_b  |    1024    |    1     |      1024       |
        | cb_r2c_in2    | CBIndex::c_5  | bfloat8_b  |    1024    |    2     |      2048       |
        | cb_c2w_mm2    | CBIndex::c_6  | bfloat8_b  |    1024    |    1     |      1024       |
        ----------------------------------------------------------------------------------------
    */

    // Define the CB configuration as a map: name -> tuple<CBIndex, DataFormat, bytes_per_tile, tiles_per_cb>
    // Note: cb_s2c_in is handled separately as it's a sharded CB
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, uint32_t, uint32_t>> cb_specs = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp8_b, 1024, 2},
        {"cb_c2c_mm0", tt::CBIndex::c_2, tt::DataFormat::Bfp8_b, 1024, 1},
        {"cb_c2c_mm1", tt::CBIndex::c_3, tt::DataFormat::Bfp8_b, 1024, 1},
        {"cb_c2w_elt", tt::CBIndex::c_4, tt::DataFormat::Bfp8_b, 1024, 1},
        {"cb_r2c_in2", tt::CBIndex::c_5, tt::DataFormat::Bfp8_b, 1024, 1},
        {"cb_c2w_mm2", tt::CBIndex::c_6, tt::DataFormat::Bfp8_b, 1024, 1}};

    [[maybe_unused]] std::map<std::string, tt::tt_metal::CBHandle> cb_handles;

    // Create sharded CB for input (cb_s2c_in at CBIndex::c_1)
    // This CB points directly to the sharded input tensor's L1 buffer
    auto cb_s2c_in_config =
        tt::tt_metal::CircularBufferConfig(input_tiles_per_core * input_tile_size, {{tt::CBIndex::c_1, input_df}})
            .set_page_size(tt::CBIndex::c_1, input_tile_size)
            .set_globally_allocated_address(*input_tensor.buffer());
    cb_handles["cb_s2c_in"] = tt::tt_metal::CreateCircularBuffer(program, cores, cb_s2c_in_config);

    // Create other CBs
    for (const auto& [name, index, dtype, bytes_per_tile, tiles_per_cb] : cb_specs) {
        auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, dtype}})
                             .set_page_size(index, bytes_per_tile);

        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);
    }

    const auto tensors = std::vector<const Tensor*>{
        &tensor_args.input_tensor,
        &tensor_args.w0_tensor,
        &tensor_args.w1_tensor,
        &tensor_args.w2_tensor,
        &tensor_args.output_tensor};

    std::vector<uint32_t> dm_compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(dm_compile_args);
    }

    // Create kernels for the program
    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/dm0.cpp",
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = dm_compile_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/dm1.cpp",
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = dm_compile_args});

    auto compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/compute.cpp",
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = true,
            .math_approx_mode = false,
            .compile_args = dm_compile_args});

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> runtime_args;
    runtime_args.push_back(0);  // Core ID placeholder
    for (const auto& tensor : tensors) {
        runtime_args.push_back(tensor->buffer()->address());
    }

    uint32_t core_id = 0;
    for (const auto& core : cores.ranges()) {
        for (const auto& core : core) {
            runtime_args[0] = core_id++;
            tt::tt_metal::SetRuntimeArgs(program, dm0_kernel_handle, core, runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, dm1_kernel_handle, core, runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, runtime_args);
        }
    }

    auto all_worker_cores_ordered =
        tensor_args.input_tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::RISCV_0_default);
    for (const auto& core : all_worker_cores_ordered) {
        log_warning(tt::LogOp, "Worker core: {}", core.str());
    }

    return cached_program_t{std::move(program), MoESharedVariables{}};
}

void MoEProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // TODO: Implement runtime argument override logic here
}

}  // namespace ttnn::operations::experimental::moe::program
