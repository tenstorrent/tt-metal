// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_program_factory.hpp"
#include "matmul_wo_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::deepseek::mla::program {

MatmulWOProgramFactory::cached_program_t MatmulWOProgramFactory::create(
    const deepseek::mla::operation_attributes_t& operation_attributes,
    const deepseek::mla::tensor_args_t& tensor_args,
    deepseek::mla::tensor_return_value_t&) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Get the cores for the program
    const auto dram_bank2core_coords =
        tensor_args.input_tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::RISCV_0_default);

    const uint32_t num_cores = dram_bank2core_coords.size();
    auto all_cores = tt::tt_metal::CoreRangeSet(dram_bank2core_coords);

    // CBs used in the Matmul WO operation
    /*
        ------------------------------------------------------------------------------------
        |     Name       |   CB Index    |   Dtype    | Tile? | Tiles/CB |  Total size (B) |
        ------------------------------------------------------------------------------------
        | cb_r2c_w0      | CBIndex::c_0  | Bfp8_b     | true  |    7*3*3 |      45696      |
        | cb_s2c_in(sh)  | CBIndex::c_1  | Float16_b  | true  |    512   |      1048576    |
        | cb_c2w_rdy     | CBIndex::c_2  | Float32    | false |    1     |      4          |
        | cb_w2c_rdy     | CBIndex::c_3  | Float32    | false |    1     |      4          |
        ------------------------------------------------------------------------------------
    */

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    // Note: cb_s2c_in is handled separately as it is sharded CB
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> cb_specs0 = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp8_b, true, 7 * 3 * 3},
        {"cb_c2w_rdy", tt::CBIndex::c_2, tt::DataFormat::Float32, false, 1},
        {"cb_w2c_rdy", tt::CBIndex::c_3, tt::DataFormat::Float32, false, 1},
    };

    [[maybe_unused]] std::map<std::string, tt::tt_metal::CBHandle> cb_handles, cb_handles_sharded;

    // Create CBs
    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : cb_specs0) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);

        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // Create sharded CBs
    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb, Buffer*
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t, tt::tt_metal::Buffer*>>
        sharded_cb_specs = {
            {"cb_s2c_in", tt::CBIndex::c_1, tt::DataFormat::Float16_b, true, 512, tensor_args.input_tensor.buffer()}};

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb, p_buffer] : sharded_cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile)
                                   .set_globally_allocated_address(*p_buffer);
        cb_handles_sharded[name] = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // Create compile args for the program
    const auto tensors =
        std::vector<const Tensor*>{&tensor_args.input_tensor, &tensor_args.w_tensor, &tensor_args.output_tensor};

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    // Create semaphores for reducing the partials at the end
    const auto reduce_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const auto collector_coord =
        tensor_args.input_tensor.device()->worker_core_from_logical_core(dram_bank2core_coords[0]);

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"layer_id", operation_attributes.layer_id},
        {"num_cores", static_cast<uint32_t>(num_cores)},
        {"collector_physical_x", static_cast<uint32_t>(collector_coord.x)},
        {"collector_physical_y", static_cast<uint32_t>(collector_coord.y)},
        {"reduce_semaphore_id", reduce_semaphore_id},
    };

    // Create kernels for the program
    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm0.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm1.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute_dummy.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> runtime_args;
    runtime_args.push_back(0);  // DRAM Bank ID placeholder
    runtime_args.push_back(0);  // VChannel placeholder
    for (const auto& tensor : tensors) {
        runtime_args.push_back(tensor->buffer()->address());
    }

    std::vector<uint32_t> vchannels;
    uint32_t dram_bank = 0;
    for (auto core : dram_bank2core_coords) {
        uint32_t vchannel = dram_bank & 0x3;

        // Check if there is any core with the same row
        auto it = std::find_if(
            dram_bank2core_coords.begin(), dram_bank2core_coords.begin() + dram_bank, [&](const auto& core_prev) {
                return core_prev.y == core.y;
            });

        // If there is any core with the same row, make sure the VChannel is different
        if (it != dram_bank2core_coords.begin() + dram_bank) {
            size_t j = std::distance(dram_bank2core_coords.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);

        runtime_args[0] = dram_bank++;
        runtime_args[1] = vchannel;

        tt::tt_metal::SetRuntimeArgs(program, dm0_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, runtime_args);
    }

    return cached_program_t{
        std::move(program),
        MatmulWOSharedVariables{
            .cb_handles_sharded = cb_handles_sharded,
            .kernel_handles = {dm0_kernel_handle, dm1_kernel_handle, compute_kernel_handle},
            .worker_cores = dram_bank2core_coords}};
}

void MatmulWOProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    deepseek::mla::tensor_return_value_t&) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    // Update sharded circular buffer addresses
    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, shared_variables.cb_handles_sharded["cb_s2c_in"], *tensor_args.input_tensor.buffer());

    // Update runtime args for all kernels with new tensor addresses
    // Runtime args layout: [3] = w_tensor address
    for (const auto& core : shared_variables.worker_cores) {
        for (const auto& kernel_handle : shared_variables.kernel_handles) {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);
            runtime_args[3] = tensor_args.w_tensor.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek::mla::program
