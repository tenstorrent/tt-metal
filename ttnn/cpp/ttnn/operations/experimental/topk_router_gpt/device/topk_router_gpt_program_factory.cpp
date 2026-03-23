// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_program_factory.hpp"
#include "topk_router_gpt_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::topk_router_gpt::program {

TopkRouterGptProgramFactory::cached_program_t TopkRouterGptProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto* device = tensor_args.input_tensor.device();

    // Get the cores for the program
    const auto dram_bank2core_coords =
        device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
    const uint32_t num_cores = dram_bank2core_coords.size();
    auto all_cores = tt::tt_metal::CoreRangeSet(dram_bank2core_coords);

    constexpr uint32_t num_groups = 4;
    constexpr uint32_t cores_per_group = 3;
    constexpr uint32_t required_cores = num_groups * cores_per_group;
    TT_FATAL(
        num_cores >= required_cores,
        "topk_router_gpt requires at least {} DRAM-aligned cores, got {}",
        required_cores,
        num_cores);

    // Tensor shapes
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    const uint32_t hidden_dim = input_shape[1];
    const uint32_t num_experts = operation_attributes.num_experts;
    constexpr uint32_t tile_hw = 32;

    const uint32_t total_k_tiles = hidden_dim / tile_hw;
    const uint32_t n_tiles = num_experts / tile_hw;
    const uint32_t k_tiles_per_core_base = total_k_tiles / cores_per_group;
    const uint32_t k_tiles_remainder = total_k_tiles % cores_per_group;
    const uint32_t max_k_tiles = k_tiles_per_core_base + (k_tiles_remainder > 0 ? 1 : 0);

    // CBs used in the topk_router_gpt operation
    // CB0-CB3 MUST have identical sizes on all cores because the DM1 kernel reads its
    // own CB2 base address and uses it as the NOC write destination for other cores.
    // If CB sizes differed between cores, the L1 offsets would diverge silently.
    /*
        ------------------------------------------------------------------------------------
        |     Name           |   CB Index    |   Dtype    | Tile? | Tiles/CB | Cores       |
        ------------------------------------------------------------------------------------
        | cb_weight          | CBIndex::c_0  | Float16_b  | true  | max_k    | All         |
        | cb_input           | CBIndex::c_1  | Float16_b  | true  | max_k    | All         |
        | cb_partial_recv    | CBIndex::c_2  | Float16_b  | true  | 2        | All         |
        | cb_local_out       | CBIndex::c_3  | Float16_b  | true  | 1        | All         |
        | cb_bias            | CBIndex::c_4  | Float16_b  | true  | 1        | Workers     |
        | cb_index           | CBIndex::c_5  | Float16_b  | true  | 1        | Workers     |
        | cb_topk_val        | CBIndex::c_6  | Float16_b  | true  | 1        | Workers     |
        | cb_gathered_val    | CBIndex::c_8  | Float16_b  | true  | 4        | Workers     |
        | cb_gathered_ind    | CBIndex::c_9  | Float16_b  | true  | 4        | Workers     |
        | cb_intermed_val    | CBIndex::c_10 | Float16_b  | true  | 2        | Collector   |
        | cb_intermed_ind    | CBIndex::c_11 | Float16_b  | true  | 1        | Collector   |
        | cb_softmax_mask    | CBIndex::c_12 | Float16_b  | true  | 1        | Collector   |
        | cb_softmax_tmp     | CBIndex::c_13 | Float16_b  | true  | 1        | Collector   |
        | cb_reduce_scalar   | CBIndex::c_14 | Float16_b  | true  | 1        | Collector   |
        | cb_bcast_scaler    | CBIndex::c_15 | Float16_b  | true  | 1        | Collector   |
        | cb_final_out       | CBIndex::c_16 | Float16_b  | true  | 2        | Collector   |
        | cb_dispatch        | CBIndex::c_19 | Float16_b  | false | var      | Collector   |
        ------------------------------------------------------------------------------------
    */

    // Create optimal ring ordering for NOC1 to minimize traffic conflicts
    // NOC1 routes: decreasing y (top) first, then decreasing x (left)
    std::vector<uint32_t> ring_pos2bank_id(num_cores);
    std::iota(ring_pos2bank_id.begin(), ring_pos2bank_id.end(), 0);

    std::sort(
        ring_pos2bank_id.begin(),
        ring_pos2bank_id.end(),
        [device, &dram_bank2core_coords](uint32_t bank_id_a, uint32_t bank_id_b) {
            const auto& pa = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_a]);
            const auto& pb = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_b]);
            if (pa.y != pb.y) {
                return pa.y > pb.y;
            }
            return pa.x > pb.x;
        });

    // Map ring positions to group roles
    const uint32_t collector_ring_pos = 2;
    const uint32_t collector_bank_id = ring_pos2bank_id[collector_ring_pos];
    const auto collector_logical = dram_bank2core_coords[collector_bank_id];
    const auto collector_physical = device->worker_core_from_logical_core(collector_logical);

    // Gather core sets by role
    std::vector<CoreCoord> sender_cores, worker_cores_vec, collector_cores_vec;
    for (uint32_t ring_pos = 0; ring_pos < required_cores; ring_pos++) {
        uint32_t bank_id = ring_pos2bank_id[ring_pos];
        uint32_t pos_in_group = ring_pos % cores_per_group;
        if (pos_in_group < 2) {
            sender_cores.push_back(dram_bank2core_coords[bank_id]);
        } else {
            worker_cores_vec.push_back(dram_bank2core_coords[bank_id]);
            if (ring_pos == collector_ring_pos) {
                collector_cores_vec.push_back(dram_bank2core_coords[bank_id]);
            }
        }
    }

    auto worker_core_set = tt::tt_metal::CoreRangeSet(worker_cores_vec);
    auto collector_core_set = tt::tt_metal::CoreRangeSet(collector_cores_vec);

    // Define CB configuration as tuples: name, CBIndex, DataFormat, is_tile, tiles_per_cb
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> all_core_cb_specs = {
        {"cb_weight", tt::CBIndex::c_0, tt::DataFormat::Float16_b, true, max_k_tiles},
        {"cb_input", tt::CBIndex::c_1, tt::DataFormat::Float16_b, true, max_k_tiles},
        {"cb_partial_recv", tt::CBIndex::c_2, tt::DataFormat::Float16_b, true, 2},
        {"cb_local_out", tt::CBIndex::c_3, tt::DataFormat::Float16_b, true, 1},
    };

    [[maybe_unused]] std::map<std::string, tt::tt_metal::CBHandle> cb_handles;

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : all_core_cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);
        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // Worker CBs (includes collector)
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> worker_cb_specs = {
        {"cb_bias", tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 1},
        {"cb_index", tt::CBIndex::c_5, tt::DataFormat::Float16_b, true, 1},
        {"cb_topk_val", tt::CBIndex::c_6, tt::DataFormat::Float16_b, true, 1},
        {"cb_gathered_val", tt::CBIndex::c_8, tt::DataFormat::Float16_b, true, num_groups},
        {"cb_gathered_ind", tt::CBIndex::c_9, tt::DataFormat::Float16_b, true, num_groups},
    };

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : worker_cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);
        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, worker_core_set, cb_config);
    }

    // Collector-only CBs
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> collector_cb_specs = {
        {"cb_intermed_val", tt::CBIndex::c_10, tt::DataFormat::Float16_b, true, 2},
        {"cb_intermed_ind", tt::CBIndex::c_11, tt::DataFormat::Float16_b, true, 1},
        {"cb_softmax_mask", tt::CBIndex::c_12, tt::DataFormat::Float16_b, true, 1},
        {"cb_softmax_tmp", tt::CBIndex::c_13, tt::DataFormat::Float16_b, true, 1},
        {"cb_reduce_scalar", tt::CBIndex::c_14, tt::DataFormat::Float16_b, true, 1},
        {"cb_bcast_scaler", tt::CBIndex::c_15, tt::DataFormat::Float16_b, true, 1},
        {"cb_final_out", tt::CBIndex::c_16, tt::DataFormat::Float16_b, true, 2},
    };

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : collector_cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);
        cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, collector_core_set, cb_config);
    }

    // CB19: Dispatch scratch (collector only, non-tile)
    uint32_t k_padded = tt::round_up(operation_attributes.k, 8);
    uint32_t dispatch_scratch_size = 2 * tile_hw * k_padded * 2;
    {
        const auto cb_config =
            tt::tt_metal::CircularBufferConfig(dispatch_scratch_size, {{tt::CBIndex::c_19, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_19, dispatch_scratch_size);
        cb_handles["cb_dispatch"] = tt::tt_metal::CreateCircularBuffer(program, collector_core_set, cb_config);
    }

    // Create compile args for the program
    const auto tensors =
        std::vector<const Tensor*>{&tensor_args.input_tensor, &tensor_args.weight_tensor, &tensor_args.bias_tensor};

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    const uint32_t tile_size_bf16 = tt::tile_size(tt::DataFormat::Float16_b);

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"num_cores", num_cores},
        {"num_groups", num_groups},
        {"cores_per_group", cores_per_group},
        {"collector_physical_x", collector_physical.x},
        {"collector_physical_y", collector_physical.y},
        {"topk_k", operation_attributes.k},
        {"k_padded", k_padded},
        {"n_tiles", n_tiles},
        {"tile_size_bf16", tile_size_bf16},
    };

    // Create kernels for the program
    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm0.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm1.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = false,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    // Create semaphores
    const uint32_t sem_partial_ready = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const uint32_t sem_topk_ready = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // VChannel computation with conflict avoidance
    std::vector<uint32_t> vchannels;
    for (uint32_t bank_id = 0; bank_id < num_cores; bank_id++) {
        uint32_t vchannel = bank_id & 0x3;
        auto it = std::find_if(
            dram_bank2core_coords.begin(), dram_bank2core_coords.begin() + bank_id, [&](const auto& core_prev) {
                return core_prev.y == dram_bank2core_coords[bank_id].y;
            });
        if (it != dram_bank2core_coords.begin() + bank_id) {
            size_t j = std::distance(dram_bank2core_coords.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);
    }

    // Dispatch output addresses
    uint32_t indices_rm_addr = std::get<0>(tensor_return_value).buffer()->address();
    uint32_t weights_rm_addr = std::get<1>(tensor_return_value).buffer()->address();
    uint32_t dispatch_aligned_page_size = std::get<0>(tensor_return_value).buffer()->aligned_page_size();

    // Set the runtime arguments for the kernels
    // Shared layout across all 3 kernels (each reads only what it needs):
    //   [0]  dram_bank_id       [1]  vchannel
    //   [2]  weight_addr        [3]  input_addr         [4]  bias_addr
    //   [5]  sem_partial_ready  [6]  is_sender           [7]  is_worker
    //   [8]  is_collector       [9]  num_k_tiles         [10] k_tile_offset
    //   [11] n_tile_id          [12] worker_phys_x       [13] worker_phys_y
    //   [14] sender_slot        [15] worker_gather_slot   [16] sem_topk_ready
    //   [17] indices_rm_addr    [18] weights_rm_addr      [19] aligned_page_size
    std::vector<uint32_t> runtime_args(20, 0);
    runtime_args[2] = tensor_args.weight_tensor.buffer()->address();
    runtime_args[3] = tensor_args.input_tensor.buffer()->address();
    runtime_args[4] = tensor_args.bias_tensor.buffer()->address();
    runtime_args[5] = sem_partial_ready;
    runtime_args[16] = sem_topk_ready;
    runtime_args[17] = indices_rm_addr;
    runtime_args[18] = weights_rm_addr;
    runtime_args[19] = dispatch_aligned_page_size;

    std::vector<CoreCoord> configured_cores;
    configured_cores.reserve(required_cores);

    for (uint32_t ring_pos = 0; ring_pos < required_cores; ring_pos++) {
        uint32_t bank_id = ring_pos2bank_id[ring_pos];
        const auto& core = dram_bank2core_coords[bank_id];
        configured_cores.push_back(core);

        uint32_t group_id = ring_pos / cores_per_group;
        uint32_t pos_in_group = ring_pos % cores_per_group;

        uint32_t k_tiles = k_tiles_per_core_base + (pos_in_group < k_tiles_remainder ? 1 : 0);
        uint32_t k_tile_offset = 0;
        for (uint32_t j = 0; j < pos_in_group; j++) {
            k_tile_offset += k_tiles_per_core_base + (j < k_tiles_remainder ? 1 : 0);
        }

        bool is_sender = (pos_in_group < 2);
        bool is_worker = (pos_in_group == 2);
        bool is_collector = (bank_id == collector_bank_id);

        uint32_t worker_ring_pos = (group_id * cores_per_group) + 2;
        uint32_t worker_bank_id_val = ring_pos2bank_id[worker_ring_pos];
        const auto worker_physical = device->worker_core_from_logical_core(dram_bank2core_coords[worker_bank_id_val]);

        runtime_args[0] = bank_id;
        runtime_args[1] = vchannels[bank_id];
        // [2-4] tensor addresses already set
        // [5] sem_partial_ready already set
        runtime_args[6] = is_sender ? 1u : 0u;
        runtime_args[7] = is_worker ? 1u : 0u;
        runtime_args[8] = is_collector ? 1u : 0u;
        runtime_args[9] = k_tiles;
        runtime_args[10] = k_tile_offset;
        runtime_args[11] = group_id;  // n_tile_id
        runtime_args[12] = worker_physical.x;
        runtime_args[13] = worker_physical.y;
        runtime_args[14] = pos_in_group;  // sender_slot
        runtime_args[15] = group_id;      // worker_gather_slot
        // [16-19] already set

        tt::tt_metal::SetRuntimeArgs(program, dm0_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_kernel_handle, core, runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, runtime_args);
    }

    return cached_program_t{
        std::move(program),
        TopkRouterGptSharedVariables{
            .kernel_handles = {dm0_kernel_handle, dm1_kernel_handle, compute_kernel_handle},
            .worker_cores = std::move(configured_cores)}};
}

void TopkRouterGptProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    // Update runtime args for all kernels with new tensor addresses
    // Runtime args layout: [2] = weight, [3] = input, [4] = bias,
    //                      [17] = indices_rm, [18] = weights_rm, [19] = aligned_page_size
    for (const auto& core : shared_variables.worker_cores) {
        for (const auto& kernel_handle : shared_variables.kernel_handles) {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);
            runtime_args[2] = tensor_args.weight_tensor.buffer()->address();
            runtime_args[3] = tensor_args.input_tensor.buffer()->address();
            runtime_args[4] = tensor_args.bias_tensor.buffer()->address();
            runtime_args[17] = std::get<0>(tensor_return_value).buffer()->address();
            runtime_args[18] = std::get<1>(tensor_return_value).buffer()->address();
            runtime_args[19] = std::get<0>(tensor_return_value).buffer()->aligned_page_size();
        }
    }
}

}  // namespace ttnn::operations::experimental::topk_router_gpt::program
