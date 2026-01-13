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
    const auto dram_adjacent_cores =
        tensor_args.input_tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::RISCV_0_default);
    for (size_t idx = 0; idx < dram_adjacent_cores.size(); ++idx) {
        auto core = dram_adjacent_cores[idx];
        log_warning(tt::LogOp, "DRAM {} mapped to core {}", idx, core.str());
    }
    auto cores = tt::tt_metal::CoreRangeSet(dram_adjacent_cores);

    // Create CBs for the program
    // CBs used in the MOE operation
    /*
        ---------------------------------------------------------------------------
        |     Name      |   CB Index    |   Dtype    | Tiles/CB |  Total size (B) |
        ---------------------------------------------------------------------------
        | cb_r2c_w0     | CBIndex::c_0  | bfp4_b     |    28    |      16128      |
        | cb_s2c_in     | CBIndex::c_1  | bfp8_b     |    224   |      243712     |
        | cb_c2c_mm0    | CBIndex::c_2  | bfp8_b     |    1     |      576        |
        | cb_c2c_mm1    | CBIndex::c_3  | bfp8_b     |    1     |      576        |
        | cb_c2w_elt    | CBIndex::c_4  | bfp8_b     |    1     |      576        |
        | cb_r2c_in2    | CBIndex::c_5  | bfp8_b     |    1     |      576        |
        | cb_c2w_out    | CBIndex::c_6  | bfp8_b     |    19    |      20672      |
        ---------------------------------------------------------------------------
    */

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    // Note: cb_s2c_in and cb_c2w_out are handled separately as they are sharded CBs
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, uint32_t>> cb_specs = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp4_b, 28},
        {"cb_c2c_mm0", tt::CBIndex::c_2, tt::DataFormat::Bfp8_b, 1},
        {"cb_c2c_mm1", tt::CBIndex::c_3, tt::DataFormat::Bfp8_b, 1},
        {"cb_c2w_elt", tt::CBIndex::c_4, tt::DataFormat::Bfp8_b, 1},
        {"cb_r2c_in2", tt::CBIndex::c_5, tt::DataFormat::Bfp8_b, 1}};

    [[maybe_unused]] std::map<std::string, tt::tt_metal::CBHandle> cb_handles;

    // Create CBs
    for (const auto& [name, index, data_format, tiles_per_cb] : cb_specs) {
        const uint32_t bytes_per_tile = tt::tile_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);

        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);
    }

    // Create sharded CBs
    // TODO(nsoraba): Add tile.get_tile_size(data_format) instead of hardcoding the tiles per CB
    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb, Buffer*
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, uint32_t, tt::tt_metal::Buffer*>>
        sharded_cb_specs = {
            {"cb_s2c_in", tt::CBIndex::c_1, tt::DataFormat::Bfp8_b, 224, tensor_args.input_tensor.buffer()},
            {"cb_c2w_out", tt::CBIndex::c_6, tt::DataFormat::Bfp8_b, 1, tensor_args.output_tensor.buffer()}};

    for (const auto& [name, index, data_format, tiles_per_cb, p_buffer] : sharded_cb_specs) {
        const uint32_t bytes_per_tile = tt::tile_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile)
                                   .set_globally_allocated_address(*p_buffer);
        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);
    }

    // Create compile args for the program
    const auto tensors = std::vector<const Tensor*>{
        &tensor_args.input_tensor,
        &tensor_args.w0_tensor,
        &tensor_args.w1_tensor,
        &tensor_args.w2_tensor,
        &tensor_args.output_tensor};

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"num_experts", operation_attributes.num_experts}};

    // Create kernels for the program
    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/dm0.cpp",
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/dm1.cpp",
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/compute_dummy.cpp",
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = true,
            .math_approx_mode = true,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> runtime_args;
    runtime_args.push_back(0);  // Core ID placeholder
    runtime_args.push_back(0);  // VChannel placeholder
    for (const auto& tensor : tensors) {
        runtime_args.push_back(tensor->buffer()->address());
    }

    std::vector<uint32_t> vcs;
    uint32_t core_id = 0;
    for (auto core : dram_adjacent_cores) {
        uint32_t vc = core_id & 0x3;

        // Check if there is any core with the same row
        auto it = std::find_if(
            dram_adjacent_cores.begin(), dram_adjacent_cores.begin() + core_id, [&](const auto& core_prev) {
                return core_prev.y == core.y;
            });

        // If there is any core with the same row, make sure the VC is different
        if (it != dram_adjacent_cores.begin() + core_id) {
            size_t j = std::distance(dram_adjacent_cores.begin(), it);
            if (vc == vcs[j]) {
                vc = (vc + 1) & 0x3;
            }
        }
        vcs.push_back(vc);

        runtime_args[0] = core_id++;
        runtime_args[1] = vc;
        tt::tt_metal::SetRuntimeArgs(program, dm0_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, runtime_args);
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
