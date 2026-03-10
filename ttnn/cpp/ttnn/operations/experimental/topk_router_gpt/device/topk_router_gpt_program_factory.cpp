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
#include <vector>

namespace ttnn::operations::experimental::topk_router_gpt {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint32_t TILE_HW = 32;
static constexpr uint32_t NUM_GROUPS = 4;       // 4 N-tiles, 4 groups
static constexpr uint32_t CORES_PER_GROUP = 3;  // 2 senders + 1 worker

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------
TopkRouterGptProgramFactory::cached_program_t TopkRouterGptProgramFactory::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto* device = tensor_args.input_tensor.device();

    // ----- Core grid setup using DRAM-aligned cores -----
    const auto dram_bank2core_coords =
        device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
    const uint32_t num_cores = dram_bank2core_coords.size();
    auto all_cores = tt::tt_metal::CoreRangeSet(dram_bank2core_coords);

    constexpr uint32_t required_cores = NUM_GROUPS * CORES_PER_GROUP;
    TT_FATAL(
        num_cores >= required_cores,
        "topk_router_gpt requires at least {} DRAM-aligned cores, got {}",
        required_cores,
        num_cores);

    // ----- Tensor shapes -----
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    const uint32_t hidden_dim = input_shape[1];
    const uint32_t num_experts = attributes.num_experts;

    const uint32_t total_k_tiles = hidden_dim / TILE_HW;
    const uint32_t n_tiles = num_experts / TILE_HW;
    const uint32_t k_tiles_per_core_base = total_k_tiles / CORES_PER_GROUP;
    const uint32_t k_tiles_remainder = total_k_tiles % CORES_PER_GROUP;
    const uint32_t max_k_tiles = k_tiles_per_core_base + (k_tiles_remainder > 0 ? 1 : 0);

    // ----- Create optimal ring ordering for NOC1 -----
    // Sort by (descending y, descending x) to minimize NOC traffic conflicts
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

    // Map ring positions to group roles:
    // Ring pos 0,1,2 = group 0 (pos 0,1 = senders, pos 2 = worker)
    // Ring pos 3,4,5 = group 1, etc.
    // Worker of group 0 (ring pos 2) = collector
    const uint32_t collector_ring_pos = 2;
    const uint32_t collector_bank_id = ring_pos2bank_id[collector_ring_pos];
    const auto collector_logical = dram_bank2core_coords[collector_bank_id];
    const auto collector_physical = device->worker_core_from_logical_core(collector_logical);

    // ----- CB Configuration -----
    // All cores get the same base CBs (CB0-CB3) so that addresses are uniform for NOC writes.
    /*
        -----------------------------------------------------------------------
        |     Name         | CB Index | Dtype      | Tiles | Cores            |
        -----------------------------------------------------------------------
        | CB_WEIGHT        | c_0      | Float16_b  | max_k | All              |
        | CB_INPUT         | c_1      | Float16_b  | max_k | All              |
        | CB_PARTIAL_RECV  | c_2      | Float16_b  | 2     | All              |
        | CB_LOCAL_OUT     | c_3      | Float16_b  | 1     | All              |
        | CB_BIAS          | c_4      | Float16_b  | 1     | Workers          |
        | CB_INDEX         | c_5      | Float16_b  | 1     | Workers          |
        | CB_TOPK_VAL      | c_6      | Float16_b  | 1     | Workers          |
        | CB_TOPK_IND      | c_7      | Float16_b  | 1     | Workers          |
        | CB_GATHERED_VAL  | c_8      | Float16_b  | 4     | Workers          |
        | CB_GATHERED_IND  | c_9      | Float16_b  | 4     | Workers          |
        | CB_INTERMED_VAL  | c_10     | Float16_b  | 2     | Collector        |
        | CB_INTERMED_IND  | c_11     | Float16_b  | 2     | Collector        |
        | CB_SOFTMAX_MASK  | c_12     | Float16_b  | 1     | Collector        |
        | CB_SOFTMAX_TMP   | c_13     | Float16_b  | 1     | Collector        |
        | CB_REDUCE_SCALAR | c_14     | Float16_b  | 1     | Collector        |
        | CB_BCAST_SCALER  | c_15     | Float16_b  | 1     | Collector        |
        | CB_FINAL_OUT     | c_16     | Float16_b  | 2     | Collector        |
        | CB_DISPATCH      | c_19     | Float16_b  | var   | Collector        |
        -----------------------------------------------------------------------
    */

    const uint32_t tile_size_bf16 = tt::tile_size(tt::DataFormat::Float16_b);

    // Gather core sets by role
    std::vector<CoreCoord> sender_cores, worker_cores, collector_cores_vec;
    for (uint32_t ring_pos = 0; ring_pos < required_cores; ring_pos++) {
        uint32_t bank_id = ring_pos2bank_id[ring_pos];
        uint32_t pos_in_group = ring_pos % CORES_PER_GROUP;
        if (pos_in_group < 2) {
            sender_cores.push_back(dram_bank2core_coords[bank_id]);
        } else {
            worker_cores.push_back(dram_bank2core_coords[bank_id]);
            if (ring_pos == collector_ring_pos) {
                collector_cores_vec.push_back(dram_bank2core_coords[bank_id]);
            }
        }
    }

    auto worker_core_set = tt::tt_metal::CoreRangeSet(worker_cores);
    auto collector_core_set = tt::tt_metal::CoreRangeSet(collector_cores_vec);

    // CB0-CB3: ALL cores (uniform sizes for consistent L1 layout)
    const std::vector<std::tuple<tt::CBIndex, uint32_t>> all_core_cbs = {
        {tt::CBIndex::c_0, max_k_tiles},  // weight
        {tt::CBIndex::c_1, max_k_tiles},  // input
        {tt::CBIndex::c_2, 2},            // partial recv
        {tt::CBIndex::c_3, 1},            // local matmul out
    };

    for (const auto& [index, num_tiles] : all_core_cbs) {
        auto cb_config =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_size_bf16, {{index, tt::DataFormat::Float16_b}})
                .set_page_size(index, tile_size_bf16);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // CB4-CB9: Worker cores (includes collector)
    const std::vector<std::tuple<tt::CBIndex, uint32_t>> worker_cbs = {
        {tt::CBIndex::c_4, 1},           // bias
        {tt::CBIndex::c_5, 1},           // index template
        {tt::CBIndex::c_6, 1},           // topk values
        {tt::CBIndex::c_7, 1},           // topk indices
        {tt::CBIndex::c_8, NUM_GROUPS},  // gathered values
        {tt::CBIndex::c_9, NUM_GROUPS},  // gathered indices
    };

    for (const auto& [index, num_tiles] : worker_cbs) {
        auto cb_config =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_size_bf16, {{index, tt::DataFormat::Float16_b}})
                .set_page_size(index, tile_size_bf16);
        tt::tt_metal::CreateCircularBuffer(program, worker_core_set, cb_config);
    }

    // CB10-CB16: Collector only
    const std::vector<std::tuple<tt::CBIndex, uint32_t>> collector_cbs = {
        {tt::CBIndex::c_10, 2},  // intermed values
        {tt::CBIndex::c_11, 2},  // intermed indices
        {tt::CBIndex::c_12, 1},  // softmax mask
        {tt::CBIndex::c_13, 1},  // softmax tmp
        {tt::CBIndex::c_14, 1},  // reduce scalar
        {tt::CBIndex::c_15, 1},  // bcast scaler
        {tt::CBIndex::c_16, 2},  // final output
    };

    for (const auto& [index, num_tiles] : collector_cbs) {
        auto cb_config =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_size_bf16, {{index, tt::DataFormat::Float16_b}})
                .set_page_size(index, tile_size_bf16);
        tt::tt_metal::CreateCircularBuffer(program, collector_core_set, cb_config);
    }

    // CB19: Dispatch scratch (collector only)
    uint32_t k_padded = tt::round_up(attributes.k, 8);
    uint32_t dispatch_scratch_size = 2 * TILE_HW * k_padded * 2;  // indices + weights
    {
        auto cb_config =
            tt::tt_metal::CircularBufferConfig(dispatch_scratch_size, {{tt::CBIndex::c_19, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_19, dispatch_scratch_size);
        tt::tt_metal::CreateCircularBuffer(program, collector_core_set, cb_config);
    }

    // ----- Compile-time args (TensorAccessorArgs pattern) -----
    const auto tensors =
        std::vector<const Tensor*>{&tensor_args.input_tensor, &tensor_args.weight_tensor, &tensor_args.bias_tensor};

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"num_cores", num_cores},
        {"num_groups", NUM_GROUPS},
        {"cores_per_group", CORES_PER_GROUP},
        {"collector_physical_x", collector_physical.x},
        {"collector_physical_y", collector_physical.y},
        {"topk_k", attributes.k},
        {"k_padded", k_padded},
        {"n_tiles", n_tiles},
        {"tile_size_bf16", tile_size_bf16},
    };

    // ----- Kernel definitions -----
    auto dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm0.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = compile_args,
            .named_compile_args = named_compile_time_args});

    auto dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm1.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
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

    // ----- Semaphores -----
    const uint32_t sem_partial_ready = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const uint32_t sem_topk_ready = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // ----- VChannel computation with conflict avoidance -----
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

    // ----- Dispatch output addresses -----
    uint32_t indices_rm_addr = std::get<0>(output_tensor).buffer()->address();
    uint32_t weights_rm_addr = std::get<1>(output_tensor).buffer()->address();
    uint32_t dispatch_aligned_page_size = std::get<0>(output_tensor).buffer()->aligned_page_size();

    // ----- Runtime arguments (per core, ring-ordered) -----
    for (uint32_t ring_pos = 0; ring_pos < required_cores; ring_pos++) {
        uint32_t bank_id = ring_pos2bank_id[ring_pos];
        const auto& core = dram_bank2core_coords[bank_id];

        uint32_t group_id = ring_pos / CORES_PER_GROUP;
        uint32_t pos_in_group = ring_pos % CORES_PER_GROUP;

        // K-tiles for this core
        uint32_t k_tiles = k_tiles_per_core_base + (pos_in_group < k_tiles_remainder ? 1 : 0);
        uint32_t k_tile_offset = 0;
        for (uint32_t j = 0; j < pos_in_group; j++) {
            k_tile_offset += k_tiles_per_core_base + (j < k_tiles_remainder ? 1 : 0);
        }

        bool is_sender = (pos_in_group < 2);
        bool is_worker = (pos_in_group == 2);
        bool is_collector = (bank_id == collector_bank_id);
        uint32_t n_tile_id = group_id;

        // Worker physical coords for this group
        uint32_t worker_ring_pos = group_id * CORES_PER_GROUP + 2;
        uint32_t worker_bank_id_val = ring_pos2bank_id[worker_ring_pos];
        const auto worker_physical = device->worker_core_from_logical_core(dram_bank2core_coords[worker_bank_id_val]);

        uint32_t sender_slot = pos_in_group;  // 0 or 1 for senders
        uint32_t worker_gather_slot = group_id;

        // DM0 args: weight + input reader
        std::vector<uint32_t> dm0_args = {
            tensor_args.weight_tensor.buffer()->address(),  // 0
            tensor_args.input_tensor.buffer()->address(),   // 1
            bank_id,                                        // 2
            vchannels[bank_id],                             // 3
            k_tiles,                                        // 4
            k_tile_offset,                                  // 5
            n_tile_id,                                      // 6
            is_worker ? 1u : 0u,                            // 7
            tensor_args.bias_tensor.buffer()->address(),    // 8
        };
        tt::tt_metal::SetRuntimeArgs(program, dm0_kernel_handle, core, dm0_args);

        // DM1 args: inter-core communication + output writer
        std::vector<uint32_t> dm1_args = {
            is_sender ? 1u : 0u,         // 0
            is_worker ? 1u : 0u,         // 1
            is_collector ? 1u : 0u,      // 2
            worker_physical.x,           // 3
            worker_physical.y,           // 4
            sem_partial_ready,           // 5
            sem_topk_ready,              // 6
            sender_slot,                 // 7
            worker_gather_slot,          // 8
            n_tile_id,                   // 9
            indices_rm_addr,             // 10
            weights_rm_addr,             // 11
            dispatch_aligned_page_size,  // 12
            vchannels[bank_id],          // 13
        };
        tt::tt_metal::SetRuntimeArgs(program, dm1_kernel_handle, core, dm1_args);

        // Compute args
        std::vector<uint32_t> compute_args = {
            is_sender ? 1u : 0u,     // 0
            is_worker ? 1u : 0u,     // 1
            is_collector ? 1u : 0u,  // 2
            k_tiles,                 // 3
            n_tile_id,               // 4
        };
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, compute_args);
    }

    return cached_program_t{
        std::move(program),
        TopkRouterGptSharedVariables{
            .kernel_handles = {dm0_kernel_handle, dm1_kernel_handle, compute_kernel_handle},
            .worker_cores = dram_bank2core_coords}};
}

// ---------------------------------------------------------------------------
// override_runtime_arguments
// ---------------------------------------------------------------------------
void TopkRouterGptProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    uint32_t indices_rm_addr = std::get<0>(output_tensor).buffer()->address();
    uint32_t weights_rm_addr = std::get<1>(output_tensor).buffer()->address();
    uint32_t dispatch_aligned_page_size = std::get<0>(output_tensor).buffer()->aligned_page_size();

    for (const auto& core : shared.worker_cores) {
        for (const auto& kernel_handle : shared.kernel_handles) {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);

            // Identify kernel by arg layout:
            // dm0: args[0] = weight_addr, args[1] = input_addr, args[8] = bias_addr
            // dm1: args[10] = indices_rm_addr, args[11] = weights_rm_addr, args[12] = aligned_page_size
            if (kernel_handle == shared.kernel_handles[0]) {
                // DM0
                runtime_args[0] = tensor_args.weight_tensor.buffer()->address();
                runtime_args[1] = tensor_args.input_tensor.buffer()->address();
                runtime_args[8] = tensor_args.bias_tensor.buffer()->address();
            } else if (kernel_handle == shared.kernel_handles[1]) {
                // DM1
                runtime_args[10] = indices_rm_addr;
                runtime_args[11] = weights_rm_addr;
                runtime_args[12] = dispatch_aligned_page_size;
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::topk_router_gpt
