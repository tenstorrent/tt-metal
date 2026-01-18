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
#include <numeric>
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
        log_info(tt::LogOp, "DRAM {} mapped to core {}", idx, core.str());
    }
    auto all_cores = tt::tt_metal::CoreRangeSet(dram_adjacent_cores);

    // Pick first 8 and next 4 cores
    auto cores_8 = tt::tt_metal::CoreRangeSet({dram_adjacent_cores.data(), 8});
    auto cores_4 = tt::tt_metal::CoreRangeSet({dram_adjacent_cores.data() + 8, 4});

    // Create CBs for the program
    // CBs used in the MOE operation
    /*
        ----------------------------------------------------------------------------
        |     Name       |   CB Index    |   Dtype    | Tiles/CB |  Total size (B) |
        ----------------------------------------------------------------------------
        | cb_r2c_w0      | CBIndex::c_0  | Bfp4_b     |    14*6  |      48384      |
        | cb_s2c_in(sh)  | CBIndex::c_1  | Float16_b  |    224*2 |      917504     |
        | cb_c2c_mm0     | CBIndex::c_2  | Float16_b  |    5|6   |      3456       |
        | cb_c2c_mm1     | CBIndex::c_3  | Float16_b  |    5|6   |      3456       |
        | cb_c2w_elt     | CBIndex::c_4  | Float16_b  |    5|6   |      3456       |
        | cb_r2c_in2     | CBIndex::c_5  | Float16_b  |    5|6   |      3456       |
        | cb_c2w_out(sh) | CBIndex::c_6  | Float16_b  |    1*2   |      4096       |
        ---------------------------------------------------------------------------
    */

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    // Note: cb_s2c_in and cb_c2w_out are handled separately as they are sharded CBs
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, uint32_t>> cb_specs0 = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp4_b, 14 * 3 * 2},
        {"cb_c2c_mm0", tt::CBIndex::c_2, tt::DataFormat::Float16_b, 5},
        {"cb_c2c_mm1", tt::CBIndex::c_3, tt::DataFormat::Float16_b, 5},
        {"cb_c2w_elt", tt::CBIndex::c_4, tt::DataFormat::Float16_b, 5},
        {"cb_r2c_in2", tt::CBIndex::c_5, tt::DataFormat::Float16_b, 72}};  // Can get 6 from other cores

    [[maybe_unused]] std::map<std::string, tt::tt_metal::CBHandle> cb_handles8, cb_handles4, cb_handles_sharded;

    // Create CBs
    for (const auto& [name, index, data_format, tiles_per_cb] : cb_specs0) {
        const uint32_t bytes_per_tile = tt::tile_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);

        cb_handles8[name] = tt::tt_metal::CreateCircularBuffer(program, cores_8, cb_config);
    }

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    // Note: cb_s2c_in and cb_c2w_out are handled separately as they are sharded CBs
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, uint32_t>> cb_specs1 = {
        {"cb_r2c_w0", tt::CBIndex::c_0, tt::DataFormat::Bfp4_b, 14 * 3 * 2},
        {"cb_c2c_mm0", tt::CBIndex::c_2, tt::DataFormat::Float16_b, 6},
        {"cb_c2c_mm1", tt::CBIndex::c_3, tt::DataFormat::Float16_b, 6},
        {"cb_c2w_elt", tt::CBIndex::c_4, tt::DataFormat::Float16_b, 6},
        {"cb_r2c_in2", tt::CBIndex::c_5, tt::DataFormat::Float16_b, 72}};

    // Create CBs
    for (const auto& [name, index, data_format, tiles_per_cb] : cb_specs1) {
        const uint32_t bytes_per_tile = tt::tile_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);

        cb_handles4[name] = tt::tt_metal::CreateCircularBuffer(program, cores_4, cb_config);
    }

    // Create sharded CBs
    // TODO(nsoraba): Add tile.get_tile_size(data_format) instead of hardcoding the tiles per CB
    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb, Buffer*
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, uint32_t, tt::tt_metal::Buffer*>>
        sharded_cb_specs = {
            {"cb_s2c_in", tt::CBIndex::c_1, tt::DataFormat::Float16_b, 224 * 2, tensor_args.input_tensor.buffer()},
            {"cb_c2w_out", tt::CBIndex::c_6, tt::DataFormat::Float16_b, 1 * 2, tensor_args.output_tensor.buffer()}};

    for (const auto& [name, index, data_format, tiles_per_cb, p_buffer] : sharded_cb_specs) {
        const uint32_t bytes_per_tile = tt::tile_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile)
                                   .set_globally_allocated_address(*p_buffer);
        cb_handles_sharded[name] = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // Create compile args for the program
    const auto tensors = std::vector<const Tensor*>{
        &tensor_args.input_tensor, &tensor_args.w0_w1_tensor, &tensor_args.w2_tensor, &tensor_args.output_tensor};

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"num_experts", operation_attributes.num_experts},
        {"layer_id", operation_attributes.layer_id},
    };

    // Create kernels for the program
    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/dm0.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/dm1.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = true,
            .math_approx_mode = true,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    // Create semaphores for ring synchronization between cores
    // Each core will have a semaphore that its predecessor will signal
    const uint32_t ring_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // Pre-compute physical coordinates for all cores (needed for NOC addressing)
    const uint32_t num_cores = dram_adjacent_cores.size();
    std::vector<CoreCoord> physical_coords;
    physical_coords.reserve(num_cores);
    for (const auto& logical_core : dram_adjacent_cores) {
        physical_coords.push_back(tensor_args.input_tensor.device()->worker_core_from_logical_core(logical_core));
    }

    // Create optimal ring ordering for NOC1 to minimize traffic conflicts
    // NOC1 routes: decreasing y (top) first, then decreasing x (left)
    // Sort cores by (descending y, descending x) to create a ring that flows naturally
    std::vector<uint32_t> ring_order(num_cores);
    std::iota(ring_order.begin(), ring_order.end(), 0);
    std::sort(ring_order.begin(), ring_order.end(), [&physical_coords](uint32_t a, uint32_t b) {
        const auto& pa = physical_coords[a];
        const auto& pb = physical_coords[b];
        if (pa.y != pb.y) {
            return pa.y > pb.y;  // Descending y
        }
        return pa.x > pb.x;  // Descending x
    });

    // Build neighbor mapping: neighbor[i] = index of neighbor for core i in the ring
    std::vector<uint32_t> neighbor_mapping(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t current_core = ring_order[i];
        uint32_t next_core = ring_order[(i + 1) % num_cores];
        neighbor_mapping[current_core] = next_core;
    }

    // Log the ring order for debugging
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t core_idx = ring_order[i];
        [[maybe_unused]] uint32_t neighbor_idx = neighbor_mapping[core_idx];
        log_debug(
            tt::LogOp,
            "Ring position {}: core {} ({},{}) -> neighbor core {} ({},{})",
            i,
            core_idx,
            physical_coords[core_idx].x,
            physical_coords[core_idx].y,
            neighbor_idx,
            physical_coords[neighbor_idx].x,
            physical_coords[neighbor_idx].y);
    }

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> runtime_args;
    runtime_args.push_back(0);  // Core ID placeholder
    runtime_args.push_back(0);  // VChannel placeholder
    for (const auto& tensor : tensors) {
        runtime_args.push_back(tensor->buffer()->address());
    }
    // Add placeholders for neighbor physical coords and semaphore
    runtime_args.push_back(0);                  // Neighbor physical x
    runtime_args.push_back(0);                  // Neighbor physical y
    runtime_args.push_back(ring_semaphore_id);  // Semaphore ID

    std::vector<uint32_t> vchannels;
    uint32_t core_id = 0;
    for (auto core : dram_adjacent_cores) {
        uint32_t vchannel = core_id & 0x3;

        // Check if there is any core with the same row
        auto it = std::find_if(
            dram_adjacent_cores.begin(), dram_adjacent_cores.begin() + core_id, [&](const auto& core_prev) {
                return core_prev.y == core.y;
            });

        // If there is any core with the same row, make sure the VChannel is different
        if (it != dram_adjacent_cores.begin() + core_id) {
            size_t j = std::distance(dram_adjacent_cores.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);

        // Use the optimized ring neighbor mapping
        const uint32_t neighbor_core_id = neighbor_mapping[core_id];
        const auto& neighbor_physical = physical_coords[neighbor_core_id];

        runtime_args[0] = core_id++;
        runtime_args[1] = vchannel;
        // Set neighbor physical coordinates
        runtime_args[6] = static_cast<uint32_t>(neighbor_physical.x);
        runtime_args[7] = static_cast<uint32_t>(neighbor_physical.y);
        // runtime_args[9] is already set to ring_semaphore_id

        tt::tt_metal::SetRuntimeArgs(program, dm0_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, runtime_args);
    }

    return cached_program_t{
        std::move(program),
        MoESharedVariables{
            .cb_handles_sharded = cb_handles_sharded,
            .kernel_handles = {dm0_kernel_handle, dm1_kernel_handle, compute_kernel_handle},
            .worker_cores = dram_adjacent_cores}};
}

void MoEProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    // Update sharded circular buffer addresses
    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, shared_variables.cb_handles_sharded["cb_s2c_in"], *tensor_args.input_tensor.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, shared_variables.cb_handles_sharded["cb_c2w_out"], *tensor_args.output_tensor.buffer());

    // Update runtime args for all kernels with new tensor addresses
    // Runtime args layout: [3] = w0_w1_tensor address, [5] = w2_tensor address
    for (const auto& core : shared_variables.worker_cores) {
        for (const auto& kernel_handle : shared_variables.kernel_handles) {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);
            runtime_args[3] = tensor_args.w0_w1_tensor.buffer()->address();
            runtime_args[4] = tensor_args.w2_tensor.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::moe::program
