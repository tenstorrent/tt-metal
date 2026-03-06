// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "tt_metal/api/tt-metalium/buffer.hpp"
#include "tt_metal/api/tt-metalium/core_coord.hpp"
#include "tt_metal/api/tt-metalium/device.hpp"
#include "tt_metal/api/tt-metalium/program.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::topk_router_gpt {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint32_t TILE_HW = 32;
static constexpr uint32_t TILE_SIZE_BF16 = 2 * 1024;  // 32x32 bf16 = 2KB
static constexpr uint32_t NUM_GROUPS = 4;             // 4 N-tiles, 4 groups
static constexpr uint32_t CORES_PER_GROUP = 3;        // 2 senders + 1 worker

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------
TopkRouterGptProgramFactory::cached_program_t TopkRouterGptProgramFactory::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    Program program{};

    auto device = tensor_args.input_tensor.device();

    // ----- Core grid setup using DRAM-aligned cores -----
    const auto dram_bank2core_coords = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::RISCV_0_default);
    const uint32_t num_cores = dram_bank2core_coords.size();  // 12 on WH
    auto all_cores = CoreRangeSet(dram_bank2core_coords);

    // ----- Tensor shapes -----
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    const uint32_t hidden_dim = input_shape[1];           // 2880
    const uint32_t num_experts = attributes.num_experts;  // 128

    const uint32_t total_k_tiles = hidden_dim / TILE_HW;  // 2880/32 = 90
    const uint32_t n_tiles = num_experts / TILE_HW;       // 128/32 = 4

    // ----- Group assignment: 4 groups of 3 cores -----
    // Group g: bank_ids [3g, 3g+1, 3g+2]
    // Within each group: cores 0,1 are senders, core 2 is the worker
    // Group 0's worker is also the collector (merges all workers' topk + softmax)

    // K-tiles distribution: each group of 3 cores needs ALL K-tiles for its N-tile.
    // Distribute 90 K-tiles among 3 cores per group (30 each).
    const uint32_t cores_per_group = CORES_PER_GROUP;  // 3
    const uint32_t k_tiles_per_core_base = total_k_tiles / cores_per_group;
    const uint32_t k_tiles_remainder = total_k_tiles % cores_per_group;
    const uint32_t max_k_tiles = k_tiles_per_core_base + (k_tiles_remainder > 0 ? 1 : 0);

    // Identify worker and collector cores
    // Worker bank_ids: 2, 5, 8, 11
    // Collector: worker of group 0 = bank_id 2
    const uint32_t collector_bank_id = 2;
    const auto collector_logical = dram_bank2core_coords[collector_bank_id];
    const auto collector_physical = device->worker_core_from_logical_core(collector_logical);

    // ----- Tensor buffer addresses -----
    auto input_buffer = tensor_args.input_tensor.buffer();
    auto weight_buffer = tensor_args.weight_tensor.buffer();
    auto bias_buffer = tensor_args.bias_tensor.buffer();

    // ----- Semaphores -----
    // sem_partial_ready: on worker cores, incremented by senders to signal partial is ready
    auto sem_partial_ready = CreateSemaphore(program, all_cores, 0);
    // sem_topk_ready: on collector core, incremented by non-collector workers
    auto sem_topk_ready = CreateSemaphore(program, all_cores, 0);

    // ----- Circular Buffers -----
    // All cores get CB0 (weight) and CB1 (input) with SAME sizes so that
    // CB2/CB3 start at the same L1 address on every core within a role.
    // Senders need: CB0, CB1, CB3 (local matmul output for DM1 to send)
    // Workers need: CB0, CB1, CB2 (partial recv), CB3 (local matmul output),
    //               CB4 (bias), CB5 (index template), CB6 (topk values), CB7 (topk indices)
    // Collector additionally: CB8-CB16 for gather, merge, softmax, output

    // Pre-compute collector CB addresses for NOC writes (needed before CB creation)
    // We'll create collector CBs first on the collector core, then reference those addresses.
    // For now, compute them analytically based on CB sizes and L1 base.

    for (uint32_t i = 0; i < num_cores; i++) {
        const auto& core = dram_bank2core_coords[i];
        CoreRangeSet single_core({CoreRange(core, core)});

        uint32_t pos_in_group = i % CORES_PER_GROUP;
        bool is_sender_core = (pos_in_group < 2);
        bool is_collector_core = (i == collector_bank_id);

        // CB0-CB3: ALL cores (same sizes for uniform L1 layout of CB2/CB3)
        // CB0: weight tiles
        auto weight_cb =
            CircularBufferConfig(max_k_tiles * TILE_SIZE_BF16, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_0, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, weight_cb);

        // CB1: input tiles
        auto input_cb =
            CircularBufferConfig(max_k_tiles * TILE_SIZE_BF16, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_1, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, input_cb);

        // CB2: partial receive (2 tiles) — sender writes here on worker via NOC
        auto partial_recv_cb = CircularBufferConfig(2 * TILE_SIZE_BF16, {{tt::CBIndex::c_2, tt::DataFormat::Float16_b}})
                                   .set_page_size(tt::CBIndex::c_2, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, partial_recv_cb);

        // CB3: local matmul output (1 tile)
        auto local_out_cb = CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_3, tt::DataFormat::Float16_b}})
                                .set_page_size(tt::CBIndex::c_3, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, local_out_cb);

        // Senders only need CB0-CB3; skip the rest
        if (is_sender_core) {
            continue;
        }

        // CB4-CB7: Worker cores (worker + collector)
        auto bias_cb = CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_4, tt::DataFormat::Float16_b}})
                           .set_page_size(tt::CBIndex::c_4, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, bias_cb);

        auto index_cb = CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_5, tt::DataFormat::Float16_b}})
                            .set_page_size(tt::CBIndex::c_5, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, index_cb);

        auto topk_val_cb = CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_6, tt::DataFormat::Float16_b}})
                               .set_page_size(tt::CBIndex::c_6, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, topk_val_cb);

        auto topk_ind_cb = CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_7, tt::DataFormat::Float16_b}})
                               .set_page_size(tt::CBIndex::c_7, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, topk_ind_cb);

        // CB8-CB9: All workers (non-collector workers need these for get_write_ptr
        // to compute collector's CB address for NOC writes)
        auto gathered_val_cb =
            CircularBufferConfig(NUM_GROUPS * TILE_SIZE_BF16, {{tt::CBIndex::c_8, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_8, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, gathered_val_cb);

        auto gathered_ind_cb =
            CircularBufferConfig(NUM_GROUPS * TILE_SIZE_BF16, {{tt::CBIndex::c_9, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_9, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, gathered_ind_cb);

        // CB10-CB16: Collector only (topk merge intermediates, softmax, output)
        if (!is_collector_core) {
            continue;
        }

        auto intermed_val_cb =
            CircularBufferConfig(2 * TILE_SIZE_BF16, {{tt::CBIndex::c_10, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_10, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, intermed_val_cb);

        auto intermed_ind_cb =
            CircularBufferConfig(2 * TILE_SIZE_BF16, {{tt::CBIndex::c_11, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_11, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, intermed_ind_cb);

        auto mask_cb = CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_12, tt::DataFormat::Float16_b}})
                           .set_page_size(tt::CBIndex::c_12, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, mask_cb);

        auto softmax_tmp_cb = CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_13, tt::DataFormat::Float16_b}})
                                  .set_page_size(tt::CBIndex::c_13, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, softmax_tmp_cb);

        auto reduce_scalar_cb =
            CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_14, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_14, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, reduce_scalar_cb);

        auto bcast_scaler_cb =
            CircularBufferConfig(1 * TILE_SIZE_BF16, {{tt::CBIndex::c_15, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_15, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, bcast_scaler_cb);

        auto final_out_cb = CircularBufferConfig(2 * TILE_SIZE_BF16, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
                                .set_page_size(tt::CBIndex::c_16, TILE_SIZE_BF16);
        CreateCircularBuffer(program, single_core, final_out_cb);

        // CB17: Scratch buffer for software untilize (ROW_MAJOR output path)
        // 32 rows × 64 cols × 2 bytes = 4096 bytes = 2 tiles
        if (attributes.untilize_output) {
            auto rm_out_cb = CircularBufferConfig(2 * TILE_SIZE_BF16, {{tt::CBIndex::c_17, tt::DataFormat::Float16_b}})
                                 .set_page_size(tt::CBIndex::c_17, TILE_SIZE_BF16);
            CreateCircularBuffer(program, single_core, rm_out_cb);
        }

        // CB19: Scratch for dispatch outputs (indices uint16 + weights bf16 in RM)
        // Size: 2 × 32 rows × k_padded × 2 bytes = 1024 bytes for k_padded=8
        uint32_t k_padded = ((attributes.k + 7) / 8) * 8;
        uint32_t dispatch_scratch_size = 2 * TILE_HW * k_padded * 2;  // indices + weights
        auto dispatch_cb = CircularBufferConfig(dispatch_scratch_size, {{tt::CBIndex::c_19, tt::DataFormat::Float16_b}})
                               .set_page_size(tt::CBIndex::c_19, dispatch_scratch_size);
        CreateCircularBuffer(program, single_core, dispatch_cb);
    }

    // ----- Kernel definitions -----
    auto dm0_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm0.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto dm1_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm1.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .math_approx_mode = false, .defines = {}});

    // ----- Runtime arguments -----
    for (uint32_t i = 0; i < num_cores; i++) {
        const auto& core = dram_bank2core_coords[i];

        uint32_t group_id = i / CORES_PER_GROUP;
        uint32_t pos_in_group = i % CORES_PER_GROUP;

        // K-tiles for this core: distribute total_k_tiles among 3 cores in the group
        uint32_t k_tiles = k_tiles_per_core_base + (pos_in_group < k_tiles_remainder ? 1 : 0);
        // K-tile offset within this group (resets per group)
        uint32_t k_tile_offset = 0;
        for (uint32_t j = 0; j < pos_in_group; j++) {
            k_tile_offset += k_tiles_per_core_base + (j < k_tiles_remainder ? 1 : 0);
        }
        bool is_sender = (pos_in_group < 2);
        bool is_worker = (pos_in_group == 2);
        bool is_collector = (i == collector_bank_id);
        uint32_t n_tile_id = group_id;  // which of 4 N-tiles this group handles

        // Worker bank_id for this group
        uint32_t worker_bank_id = group_id * CORES_PER_GROUP + 2;
        const auto worker_logical = dram_bank2core_coords[worker_bank_id];
        const auto worker_physical = device->worker_core_from_logical_core(worker_logical);

        // DRAM bank for this core
        uint32_t dram_bank_id = i;

        // Virtual channel
        uint32_t vchannel = 0;
        for (uint32_t j = 0; j < i; j++) {
            if (dram_bank2core_coords[j] == core) {
                vchannel++;
            }
        }

        // DM0 args: weight + input reader
        std::vector<uint32_t> dm0_args = {
            weight_buffer->address(),  // 0
            input_buffer->address(),   // 1
            dram_bank_id,              // 2
            vchannel,                  // 3
            k_tiles,                   // 4
            k_tile_offset,             // 5
            n_tile_id,                 // 6
            TILE_SIZE_BF16,            // 7
            n_tiles,                   // 8: total N-tiles (4, for weight tile indexing)
            is_worker ? 1u : 0u,       // 9
            bias_buffer->address(),    // 10
        };
        SetRuntimeArgs(program, dm0_kernel_id, core, dm0_args);

        // DM1 args: inter-core communication + output writer
        // Sender slot_id: pos_in_group (0 or 1) — which slot in worker's CB2
        uint32_t sender_slot = pos_in_group;  // 0 or 1 for senders, unused for workers
        // Worker slot in collector's gathered CB: group_id (0-3)
        uint32_t worker_gather_slot = group_id;

        // Dispatch output addresses (indices_rm + weights_rm)
        uint32_t k_padded = ((attributes.k + 7) / 8) * 8;
        // Use buffer's aligned_page_size for InterleavedAddrGen — the DRAM allocator
        // pads pages to DRAM_ALIGNMENT (32 bytes on WH), so the kernel must use
        // the aligned stride, not the raw data size.
        uint32_t indices_rm_addr = std::get<0>(output_tensor).buffer()->address();
        uint32_t weights_rm_addr = std::get<1>(output_tensor).buffer()->address();
        uint32_t dispatch_aligned_page_size = std::get<0>(output_tensor).buffer()->aligned_page_size();

        std::vector<uint32_t> dm1_args = {
            is_sender ? 1u : 0u,                     // 0
            is_worker ? 1u : 0u,                     // 1
            is_collector ? 1u : 0u,                  // 2
            worker_physical.x,                       // 3
            worker_physical.y,                       // 4
            collector_physical.x,                    // 5
            collector_physical.y,                    // 6
            sem_partial_ready,                       // 7
            sem_topk_ready,                          // 8
            TILE_SIZE_BF16,                          // 9
            0u,                                      // 10: output_addr (unused - dispatch outputs only)
            attributes.k,                            // 11: topk_k
            sender_slot,                             // 12: sender's slot in worker's CB2
            worker_gather_slot,                      // 13: worker's slot in collector's CB8/CB9
            n_tile_id,                               // 14
            static_cast<uint32_t>(NUM_GROUPS),       // 15
            attributes.untilize_output ? 1u : 0u,    // 16: untilize output flag
            attributes.untilize_output ? 128u : 0u,  // 17: RM page size (64 bf16 = 128 bytes)
            1u,                                      // 18: produce dispatch outputs (always enabled)
            indices_rm_addr,                         // 19: indices RM buffer address
            weights_rm_addr,                         // 20: weights RM buffer address
            k_padded,                                // 21: k padded to multiple of 8
            dispatch_aligned_page_size,              // 22: DRAM-aligned page size for dispatch outputs
        };
        SetRuntimeArgs(program, dm1_kernel_id, core, dm1_args);

        // Compute args
        std::vector<uint32_t> compute_args = {
            is_sender ? 1u : 0u,                // 0
            is_worker ? 1u : 0u,                // 1
            is_collector ? 1u : 0u,             // 2
            k_tiles,                            // 3
            n_tile_id,                          // 4
            static_cast<uint32_t>(NUM_GROUPS),  // 5
            attributes.k,                       // 6: topk_k
        };
        SetRuntimeArgs(program, compute_kernel_id, core, compute_args);
    }

    return {std::move(program), {dm0_kernel_id, dm1_kernel_id, compute_kernel_id, all_cores, num_cores}};
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

    auto device = tensor_args.input_tensor.device();
    const auto dram_bank2core_coords = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::RISCV_0_default);

    auto input_buffer = tensor_args.input_tensor.buffer();
    auto weight_buffer = tensor_args.weight_tensor.buffer();
    auto bias_buffer = tensor_args.bias_tensor.buffer();

    // Dispatch output addresses (indices_rm + weights_rm)
    uint32_t indices_rm_addr = std::get<0>(output_tensor).buffer()->address();
    uint32_t weights_rm_addr = std::get<1>(output_tensor).buffer()->address();
    uint32_t dispatch_aligned_page_size = std::get<0>(output_tensor).buffer()->aligned_page_size();

    for (uint32_t i = 0; i < shared.num_cores; i++) {
        const auto& core = dram_bank2core_coords[i];

        auto& dm0_args = GetRuntimeArgs(program, shared.dm0_kernel_id, core);
        dm0_args[0] = weight_buffer->address();
        dm0_args[1] = input_buffer->address();
        dm0_args[10] = bias_buffer->address();

        auto& dm1_args = GetRuntimeArgs(program, shared.dm1_kernel_id, core);
        dm1_args[19] = indices_rm_addr;
        dm1_args[20] = weights_rm_addr;
        dm1_args[22] = dispatch_aligned_page_size;
    }
}

}  // namespace ttnn::operations::experimental::topk_router_gpt
