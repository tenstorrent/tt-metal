// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gate_mm_program_factory.hpp"
#include "moe_gate_mm_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm::program {

MoEGateMMProgramFactory::cached_program_t MoEGateMMProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t&) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Get the cores for the program
    const auto dram_bank2core_coords =
        tensor_args.input_tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::RISCV_0_default);

    const uint32_t num_cores = dram_bank2core_coords.size();
    auto all_cores = tt::tt_metal::CoreRangeSet(dram_bank2core_coords);

    // CBs used in the MoE Gate MM operation
    /*
        ------------------------------------------------------------------------------------
        |     Name       |   CB Index    |   Dtype    | Tile? | Tiles/CB |  Total size (B) |
        ------------------------------------------------------------------------------------
        | cb_r2c_w       | CBIndex::c_0  | Float16_b  | true  |    32*3  |      196608     |
        | cb_s2c_in(sh)  | CBIndex::c_1  | Float16_b  | true  |    224   |      458752     |
        | cb_c2w_rdy     | CBIndex::c_2  | Float32    | false |    1     |      4          |
        | cb_w2c_in2     | CBIndex::c_3  | Float32    | true  |    1     |      2048       |
        | cb_s2c_out(sh) | CBIndex::c_4  | Float16_b  | true  |    1     |      2048       |
        | cb_w2c_in3     | CBIndex::c_5  | Float16_b  | true  |    1     |      2048       |
        | cb_w2c_in4     | CBIndex::c_6  | Float16_b  | true  |    1     |      2048       |
        | cb_w2c_in5     | CBIndex::c_7  | Float16_b  | true  |    1     |      2048       |
        | cb_w2c_in6     | CBIndex::c_8  | Float16_b  | true  |    4     |      8192       |
        | cb_w2c_in7     | CBIndex::c_9  | Float16_b  | true  |    4     |      8192      |
        ------------------------------------------------------------------------------------
    */

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> cb_specs0 = {
        {"cb_r2c_w", tt::CBIndex::c_0, tt::DataFormat::Float16_b, true, 32 * 3},
        {"cb_c2w_rdy", tt::CBIndex::c_2, tt::DataFormat::Float32, false, 1},
        {"cb_w2c_in2", tt::CBIndex::c_3, tt::DataFormat::Float32, true, 1},
        {"cb_w2c_in3", tt::CBIndex::c_5, tt::DataFormat::Float16_b, true, 1},
        {"cb_w2c_in4", tt::CBIndex::c_6, tt::DataFormat::Float16_b, true, 1},
        {"cb_w2c_in5", tt::CBIndex::c_7, tt::DataFormat::Float16_b, true, 1},
        {"cb_w2c_in6", tt::CBIndex::c_8, tt::DataFormat::Float16_b, true, 4},
        {"cb_w2c_in7", tt::CBIndex::c_9, tt::DataFormat::Float16_b, true, 4},
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
            {"cb_s2c_in", tt::CBIndex::c_1, tt::DataFormat::Float16_b, true, 224, tensor_args.input_tensor.buffer()},
            {"cb_s2c_out", tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 1, tensor_args.output_tensor.buffer()}};

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

    // Create optimal ring ordering for NOC1 to minimize traffic conflicts
    // NOC1 routes: decreasing y (top) first, then decreasing x (left)
    // Sort cores by (descending y, descending x) to create a ring that flows naturally
    std::vector<uint32_t> ring_pos2bank_id(num_cores);
    std::iota(ring_pos2bank_id.begin(), ring_pos2bank_id.end(), 0);
    auto *device = tensor_args.input_tensor.device();

    std::sort(
        ring_pos2bank_id.begin(),
        ring_pos2bank_id.end(),
        [device, &dram_bank2core_coords](uint32_t bank_id_a, uint32_t bank_id_b) {
            const auto& pa = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_a]);
            const auto& pb = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_b]);
            if (pa.y != pb.y) {
                return pa.y > pb.y;  // Descending y
            }
            return pa.x > pb.x;  // Descending x
        });

    // For every third core, figure out the physical coords of the two cores after it.
    std::unordered_map<uint32_t, std::array<uint32_t, 5>> dram_bank2neighbors;
    for (uint32_t ring_pos = 0; ring_pos < num_cores; ring_pos += 3) {
        auto bank_id = ring_pos2bank_id[ring_pos];
        auto bank_id_next1 = ring_pos2bank_id[ring_pos + 1];
        auto bank_id_next2 = ring_pos2bank_id[ring_pos + 2];

        const auto& core_next1 = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_next1]);
        const auto& core_next2 = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_next2]);

        dram_bank2neighbors[bank_id] = {1, core_next1.x, core_next1.y, core_next2.x, core_next2.y};
    }

    // We also need the reverse mapping for bank_id to N tile_id
    std::unordered_map<uint32_t, uint32_t> bank2tile_id;
    uint32_t tile_id = 0;
    for (uint32_t core_id = 0; core_id < num_cores; core_id++) {
        if ((core_id % 3) == 0) {
            continue;
        }
        bank2tile_id[ring_pos2bank_id[core_id]] = tile_id++;
    }

    const auto& collector_core = device->worker_core_from_logical_core(dram_bank2core_coords[ring_pos2bank_id[11]]);
    const auto& first_core = device->worker_core_from_logical_core(dram_bank2core_coords[ring_pos2bank_id[0]]);

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"layer_id", operation_attributes.layer_id},
        {"num_cores", static_cast<uint32_t>(num_cores)},
        {"collector_physical_x", collector_core.x},
        {"collector_physical_y", collector_core.y},
        {"first_physical_x", first_core.x},
        {"first_physical_y", first_core.y},
        {"column_id", operation_attributes.column_id},
    };

    // Create kernels for the program
    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/dm0.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/dm1.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    // Create semaphores to wait for the partial to arrive from the other core
    // There will be 8 cores, each waiting for partial to come from 4 other cores.
    // The 4 cores will send partial to two cores each.
    const uint32_t partial_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const uint32_t raw_scores_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> runtime_args;
    runtime_args.push_back(0);  // DRAM Bank ID placeholder
    runtime_args.push_back(0);  // VChannel placeholder

    for (const auto& tensor : tensors) {
        runtime_args.push_back(tensor->buffer()->address());
    }

    // Add placeholders for neighbor physical coords and semaphore
    runtime_args.push_back(partial_semaphore_id);
    runtime_args.push_back(0);  // If this is a sender core
    runtime_args.push_back(0);  // Neighbor1 physical x
    runtime_args.push_back(0);  // Neighbor1 physical y
    runtime_args.push_back(0);  // Neighbor2 physical x
    runtime_args.push_back(0);  // Neighbor2 physical y
    runtime_args.push_back(0);  // Core ID
    runtime_args.push_back(raw_scores_semaphore_id);

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

        runtime_args[0] = dram_bank;
        runtime_args[1] = vchannel;
        // runtime_args[2-4] are already set to tensor addresses
        // runtime_args[5] is already set to reduce_semaphore
        // Do this only if the key exists in the map.
        if (dram_bank2neighbors.contains(dram_bank)) {
            runtime_args[6] = dram_bank2neighbors[dram_bank][0];
            runtime_args[7] = dram_bank2neighbors[dram_bank][1];
            runtime_args[8] = dram_bank2neighbors[dram_bank][2];
            runtime_args[9] = dram_bank2neighbors[dram_bank][3];
            runtime_args[10] = dram_bank2neighbors[dram_bank][4];
            runtime_args[11] = 0;
        } else {
            runtime_args[6] = 0;
            runtime_args[7] = 0;
            runtime_args[8] = 0;
            runtime_args[9] = 0;
            runtime_args[10] = 0;
            runtime_args[11] = bank2tile_id[dram_bank];
        }

        tt::tt_metal::SetRuntimeArgs(program, dm0_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, runtime_args);

        dram_bank++;
    }

    return cached_program_t{
        std::move(program),
        MoEGateMMSharedVariables{
            .cb_handles_sharded = cb_handles_sharded,
            .kernel_handles = {dm0_kernel_handle, dm1_kernel_handle, compute_kernel_handle},
            .worker_cores = dram_bank2core_coords}};
}

void MoEGateMMProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t&) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    // Update sharded circular buffer addresses
    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, shared_variables.cb_handles_sharded["cb_s2c_in"], *tensor_args.input_tensor.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, shared_variables.cb_handles_sharded["cb_s2c_out"], *tensor_args.output_tensor.buffer());

    // Update runtime args for all kernels with new tensor addresses
    // Runtime args layout: [2] = input_tensor address, [3] = w_tensor address, [4] = output_tensor address
    for (const auto& core : shared_variables.worker_cores) {
        for (const auto& kernel_handle : shared_variables.kernel_handles) {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);
            runtime_args[3] = tensor_args.w_tensor.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm::program
