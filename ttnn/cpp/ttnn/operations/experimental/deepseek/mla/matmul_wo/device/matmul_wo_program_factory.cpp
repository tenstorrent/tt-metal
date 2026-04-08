// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_program_factory.hpp"
#include "matmul_wo_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::deepseek::mla::program {

static std::vector<CoreCoord> find_collector_core_coords(
    const CoreCoord& full_grid_size, const std::set<CoreCoord>& dram_cores_set, size_t num_collectors) {
    std::vector<CoreCoord> collector_core_coords;
    for (int32_t y = full_grid_size.y - 1; y >= 0; --y) {
        for (int32_t x = full_grid_size.x - 1; x >= 0; --x) {
            const auto core_coord = CoreCoord(x, y);
            if (!dram_cores_set.contains(core_coord)) {
                collector_core_coords.push_back(core_coord);
                if (collector_core_coords.size() == num_collectors) {
                    return collector_core_coords;
                }
            }
        }
    }

    // We come here if we did not find enough collector cores; TT_FATAL is expected not to return.
    TT_FATAL(false, "Failed to find {} cores", num_collectors);
    return {};
}

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
    auto dram_cores = CoreRangeSet(dram_bank2core_coords);

    // Let us find 7 cores that are not taken by the DRAM reader/writer kernels
    const auto full_grid_size = tensor_args.input_tensor.device()->compute_with_storage_grid_size();

    // Let us create a set of DRAM cores so we can do an easy find operation
    const auto dram_cores_set = std::set<CoreCoord>(dram_bank2core_coords.begin(), dram_bank2core_coords.end());

    const auto collector_core_coords = find_collector_core_coords(full_grid_size, dram_cores_set, 7);
    const auto collector_cores = CoreRangeSet(collector_core_coords);

    // Convert the collector core coordinates to physical coordinates
    std::vector<uint32_t> collector_core_physical_coords;
    for (const auto& core_coord : collector_core_coords) {
        const auto physical_core_coord = tensor_args.input_tensor.device()->worker_core_from_logical_core(core_coord);
        collector_core_physical_coords.push_back(physical_core_coord.x);
        collector_core_physical_coords.push_back(physical_core_coord.y);
    }

    // Put them in defines for kernel to access
    const std::map<std::string, std::string> kernel_defines = {
        {"COLLECTOR_CORE_COORDS", ttnn::operations::ccl::common::stringify(collector_core_physical_coords)}};

    auto all_cores = dram_cores.merge(collector_cores);

    // CBs used in the Matmul WO operation
    /*
        ------------------------------------------------------------------------------------
        |     Name       |   CB Index    |   Dtype    | Tile? | Tiles/CB |  Total size (B) |
        ------------------------------------------------------------------------------------
        | cb_r2c_w       | CBIndex::c_0  | Bfp8_b     | true  |    7*3*2 |      45696      |
        | cb_s2c_in(sh)  | CBIndex::c_1  | Float16_b  | true  |    512   |      1048576    |
        | cb_c2w_out     | CBIndex::c_2  | Float16_b  | true  |    28    |      57344      |
        | cb_s2c_in2     | CBIndex::c_3  | Float16_b  | true  |    48    |      98304      |
        | cb_s2c_out(sh) | CBIndex::c_4  | Float16_b  | true  |    4     |      8192       |
        ------------------------------------------------------------------------------------
    */

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t, CoreRangeSet>> cb_specs0 = {
        {"cb_r2c_w", tt::CBIndex::c_0, tt::DataFormat::Bfp8_b, true, 7 * 3 * 2, dram_cores},
        {"cb_c2w_out", tt::CBIndex::c_2, tt::DataFormat::Float16_b, true, 28, dram_cores},
        {"cb_s2c_in2", tt::CBIndex::c_3, tt::DataFormat::Float16_b, true, 48, all_cores},
    };

    [[maybe_unused]] std::map<std::string, tt::tt_metal::CBHandle> cb_handles, cb_handles_sharded;

    // Create CBs
    for (const auto& [name, index, data_format, is_tile, tiles_per_cb, core_range_set] : cb_specs0) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);

        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);
    }

    // Create sharded CBs
    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb, Buffer*
    const std::vector<
        std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t, tt::tt_metal::Buffer*, CoreRangeSet>>
        sharded_cb_specs = {
            {"cb_s2c_in",
             tt::CBIndex::c_1,
             tt::DataFormat::Float16_b,
             true,
             512,
             tensor_args.input_tensor.buffer(),
             dram_cores},
            {"cb_s2c_out",
             tt::CBIndex::c_4,
             tt::DataFormat::Float16_b,
             true,
             4,
             tensor_args.output_tensor.buffer(),
             collector_cores}};

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb, p_buffer, core_range_set] : sharded_cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile)
                                   .set_globally_allocated_address(*p_buffer);
        cb_handles_sharded[name] = tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);
    }

    // Create compile args for the program
    const auto tensors =
        std::vector<const Tensor*>{&tensor_args.input_tensor, &tensor_args.w_tensor, &tensor_args.output_tensor};

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    // Create semaphores for reducing the partials at the end
    const auto reduce_semaphore_id = tt::tt_metal::CreateSemaphore(program, collector_cores, 0);

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"layer_id", operation_attributes.layer_id},
        {"num_cores", static_cast<uint32_t>(num_cores)},
        {"reduce_semaphore_id", reduce_semaphore_id},
    };

    // Create kernels for the program
    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm0.cpp",
        dram_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = compile_args,
            .defines = kernel_defines,
            .named_compile_args = named_compile_time_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm1.cpp",
        dram_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = compile_args,
            .defines = kernel_defines,
            .named_compile_args = named_compile_time_args});

    auto compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp",
        dram_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .compile_args = compile_args,
            .defines = kernel_defines,
            .named_compile_args = named_compile_time_args});

    //-------------------------------------------------------------------------
    // Collector cores - these collect all data and reduce them.
    //-------------------------------------------------------------------------
    auto dm1_collector_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm1_collector.cpp",
        collector_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = compile_args,
            .defines = kernel_defines,
            .named_compile_args = named_compile_time_args});

    auto compute_collector_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute_collector.cpp",
        collector_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .compile_args = compile_args,
            .defines = kernel_defines,
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

    std::vector<uint32_t> collector_runtime_args;
    collector_runtime_args.push_back(0);  // Core ID placeholder

    uint32_t core_id = 0;
    for (const auto& core : collector_core_coords) {
        collector_runtime_args[0] = core_id++;
        tt::tt_metal::SetRuntimeArgs(program, dm1_collector_kernel_handle, core, collector_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_collector_kernel_handle, core, collector_runtime_args);
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

    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, shared_variables.cb_handles_sharded["cb_s2c_out"], *tensor_args.output_tensor.buffer());

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
