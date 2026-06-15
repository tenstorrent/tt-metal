// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_program_factory.hpp"
#include "topk_router_gpt_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::topk_router_gpt::program {

using namespace tt::tt_metal;

ProgramDescriptor TopkRouterGptProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
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

    ProgramDescriptor desc;

    // Define CB configuration as tuples: name, CBIndex, DataFormat, is_tile, tiles_per_cb
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> all_core_cb_specs = {
        {"cb_weight", tt::CBIndex::c_0, tt::DataFormat::Float16_b, true, max_k_tiles},
        {"cb_input", tt::CBIndex::c_1, tt::DataFormat::Float16_b, true, max_k_tiles},
        {"cb_partial_recv", tt::CBIndex::c_2, tt::DataFormat::Float16_b, true, 2},
        {"cb_local_out", tt::CBIndex::c_3, tt::DataFormat::Float16_b, true, 1},
    };

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : all_core_cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles_per_cb * bytes_per_tile,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = data_format,
                .page_size = bytes_per_tile,
            }}},
        });
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
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles_per_cb * bytes_per_tile,
            .core_ranges = worker_core_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = data_format,
                .page_size = bytes_per_tile,
            }}},
        });
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
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles_per_cb * bytes_per_tile,
            .core_ranges = collector_core_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = data_format,
                .page_size = bytes_per_tile,
            }}},
        });
    }

    // CB19: Dispatch scratch (collector only, non-tile)
    uint32_t k_padded = tt::round_up(operation_attributes.k, 8);
    uint32_t dispatch_scratch_size = 2 * tile_hw * k_padded * 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = dispatch_scratch_size,
        .core_ranges = collector_core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_19,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = dispatch_scratch_size,
        }}},
    });

    // Create compile args for the program
    const auto tensors =
        std::vector<const Tensor*>{&tensor_args.input_tensor, &tensor_args.weight_tensor, &tensor_args.bias_tensor};

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }
    // Output tensor accessors (indices_rm and weights_rm)
    tt::tt_metal::TensorAccessorArgs(*std::get<0>(tensor_return_value).buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(*std::get<1>(tensor_return_value).buffer()).append_to(compile_args);

    const uint32_t tile_size_bf16 = tt::tile_size(tt::DataFormat::Float16_b);

    KernelDescriptor::NamedCompileTimeArgs named_compile_time_args = {
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
    KernelDescriptor dm0_desc{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm0.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .config =
            DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_0,
            },
    };

    KernelDescriptor dm1_desc{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm1.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .config =
            DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_1,
            },
    };

    KernelDescriptor compute_desc{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/compute.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .config =
            ComputeConfigDescriptor{
                .math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .bfp8_pack_precise = false,
                .math_approx_mode = false,
            },
    };

    // Create semaphores
    const uint32_t sem_partial_ready = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_partial_ready,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });
    const uint32_t sem_topk_ready = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_topk_ready,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

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

    // Buffer pointers for runtime args
    auto* weight_buffer = tensor_args.weight_tensor.buffer();
    auto* input_buffer = tensor_args.input_tensor.buffer();
    auto* bias_buffer = tensor_args.bias_tensor.buffer();
    auto* indices_rm_buffer = std::get<0>(tensor_return_value).buffer();
    auto* weights_rm_buffer = std::get<1>(tensor_return_value).buffer();
    uint32_t dispatch_aligned_page_size = indices_rm_buffer->aligned_page_size();

    // Set the runtime arguments for the kernels
    // Shared layout across all 3 kernels (each reads only what it needs):
    //   [0]  dram_bank_id       [1]  vchannel
    //   [2]  weight_addr        [3]  input_addr         [4]  bias_addr
    //   [5]  sem_partial_ready  [6]  is_sender           [7]  is_worker
    //   [8]  is_collector       [9]  num_k_tiles         [10] k_tile_offset
    //   [11] n_tile_id          [12] worker_phys_x       [13] worker_phys_y
    //   [14] sender_slot        [15] worker_gather_slot   [16] sem_topk_ready
    //   [17] indices_rm_addr    [18] weights_rm_addr      [19] aligned_page_size

    for (uint32_t ring_pos = 0; ring_pos < required_cores; ring_pos++) {
        uint32_t bank_id = ring_pos2bank_id[ring_pos];
        const auto& core = dram_bank2core_coords[bank_id];

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

        KernelDescriptor::RTArgList rt_args;
        rt_args.reserve(20);
        rt_args.push_back(bank_id);                                   // [0] dram_bank_id
        rt_args.push_back(vchannels[bank_id]);                        // [1] vchannel
        rt_args.push_back(weight_buffer);                             // [2] weight_addr
        rt_args.push_back(input_buffer);                              // [3] input_addr
        rt_args.push_back(bias_buffer);                               // [4] bias_addr
        rt_args.push_back(sem_partial_ready);                         // [5] sem_partial_ready
        rt_args.push_back(is_sender ? 1u : 0u);                       // [6] is_sender
        rt_args.push_back(is_worker ? 1u : 0u);                       // [7] is_worker
        rt_args.push_back(is_collector ? 1u : 0u);                    // [8] is_collector
        rt_args.push_back(k_tiles);                                   // [9] num_k_tiles
        rt_args.push_back(k_tile_offset);                             // [10] k_tile_offset
        rt_args.push_back(group_id);                                  // [11] n_tile_id
        rt_args.push_back(static_cast<uint32_t>(worker_physical.x));  // [12] worker_phys_x
        rt_args.push_back(static_cast<uint32_t>(worker_physical.y));  // [13] worker_phys_y
        rt_args.push_back(pos_in_group);                              // [14] sender_slot
        rt_args.push_back(group_id);                                  // [15] worker_gather_slot
        rt_args.push_back(sem_topk_ready);                            // [16] sem_topk_ready
        rt_args.push_back(indices_rm_buffer);                         // [17] indices_rm_addr
        rt_args.push_back(weights_rm_buffer);                         // [18] weights_rm_addr
        rt_args.push_back(dispatch_aligned_page_size);                // [19] aligned_page_size

        dm0_desc.emplace_runtime_args(core, rt_args);
        dm1_desc.emplace_runtime_args(core, rt_args);
        compute_desc.emplace_runtime_args(core, rt_args);
    }

    desc.kernels.push_back(std::move(dm0_desc));
    desc.kernels.push_back(std::move(dm1_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::topk_router_gpt::program
