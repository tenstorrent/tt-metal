// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_moe_compute_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt_metal/api/tt-metalium/buffer.hpp"
#include "tt_metal/api/tt-metalium/core_coord.hpp"
#include "tt_metal/api/tt-metalium/device.hpp"
#include "tt_metal/api/tt-metalium/mesh_device.hpp"
#include "tt_metal/api/tt-metalium/program.hpp"

// Fabric includes (used when enable_fabric_return is true on multi-device)
#include <tt-metalium/experimental/fabric/fabric.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::prefill_moe_compute {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint32_t TILE_HW = 32;
static constexpr uint32_t BF16_TILE_BYTES = 2 * 1024;
static constexpr uint32_t BFP4_TILE_BYTES = 576;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
static uint32_t compute_n_block(uint32_t n_per_core, uint32_t max_block = 7) {
    uint32_t limit = std::min(n_per_core, max_block);
    for (uint32_t b = limit; b >= 1; --b) {
        if (n_per_core % b == 0) {
            return b;
        }
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Kernel paths
// ---------------------------------------------------------------------------
static const std::string READER_KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/prefill_moe_compute/device/kernels/expert_reader_multi.cpp";
static const std::string WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/prefill_moe_compute/device/kernels/expert_writer_multi.cpp";
static const std::string COMPUTE_KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/prefill_moe_compute/device/kernels/compute_expert_multi.cpp";
static const std::string DISPATCH_KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/prefill_moe_compute/device/kernels/dispatch_writer_fused.cpp";
static const std::string COMBINE_KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/prefill_moe_compute/device/kernels/combine_dm_fused.cpp";
static const std::string RETURN_KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/prefill_moe_compute/device/kernels/fabric_return.cpp";
static const std::string RECV_KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/prefill_moe_compute/device/kernels/recv_accumulate.cpp";

// ---------------------------------------------------------------------------
// create_at — per-device program creation for MeshWorkloadFactoryConcept
// ---------------------------------------------------------------------------
ttnn::device_operation::CachedProgram<PrefillMoeComputeMeshFactory::shared_variables_t>
PrefillMoeComputeMeshFactory::create_at(
    const operation_attributes_t& attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output_tensor*/) {
    Program program{};

    auto* device = tensor_args.hidden_states.device();
    const uint32_t num_experts = attributes.num_experts;
    const uint32_t num_cores = attributes.num_cores;
    const uint32_t grid_x = attributes.grid_x;
    const uint32_t grid_y = attributes.grid_y;
    const bool enable_fabric_return = attributes.enable_fabric_return;

    // ----- Per-device metadata lookup -----
    const bool need_mesh = (attributes.per_device_combine_metadata.size() > 1 || enable_fabric_return);
    uint32_t device_index = 0;
    tt::tt_metal::distributed::MeshDevice* mesh_dev = nullptr;
    if (need_mesh) {
        mesh_dev = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
        TT_FATAL(mesh_dev != nullptr, "Multi-device metadata or fabric return requires a MeshDevice");
        device_index = mesh_coordinate.to_linear_index(mesh_dev->shape());
    }
    TT_FATAL(
        device_index < attributes.per_device_combine_metadata.size(),
        "device_index {} out of range for per_device_combine_metadata of size {}",
        device_index,
        attributes.per_device_combine_metadata.size());
    const auto& combine_metadata = attributes.per_device_combine_metadata[device_index];

    // Determine if this device needs fabric sends (multi-device return)
    const bool fabric_send_enabled =
        enable_fabric_return && mesh_dev != nullptr && attributes.return_metadata.size() > 1;

    // ----- Derive tile dimensions from tensor shapes -----
    const auto& hs_shape = tensor_args.hidden_states.logical_shape();
    const uint32_t D = hs_shape[3];
    const uint32_t D_tiles = D / TILE_HW;

    const auto& gu_shape = tensor_args.gate_up_weights[0].logical_shape();
    const uint32_t D_FF = gu_shape[3];

    const uint32_t k_tiles = D_tiles;
    const uint32_t n_weight_tiles_gu = D_FF / TILE_HW;
    const uint32_t n_weight_per_core_gu = n_weight_tiles_gu / num_cores;
    const uint32_t n_out_per_core = n_weight_per_core_gu / 2;

    const auto& dn_shape = tensor_args.down_weights[0].logical_shape();
    const uint32_t D_FF_HALF_padded = dn_shape[2];
    const uint32_t k_tiles_dn = D_FF_HALF_padded / TILE_HW;
    const uint32_t n_tiles_dn = D_tiles;
    const uint32_t n_per_core_dn = n_tiles_dn / num_cores;

    const uint32_t N_BLOCK_GU = compute_n_block(n_weight_per_core_gu);
    const uint32_t N_BLOCK_DN = compute_n_block(n_per_core_dn);

    // ----- Core grids -----
    CoreRange compute_range(CoreCoord(0, 0), CoreCoord(grid_x - 1, grid_y - 1));
    CoreRangeSet compute_cores(std::vector<CoreRange>{compute_range});

    CoreCoord dispatch_core_logical(0, grid_y);
    CoreRange dispatch_range(dispatch_core_logical, dispatch_core_logical);
    CoreRangeSet dispatch_cores(std::vector<CoreRange>{dispatch_range});

    CoreCoord combine_core_logical(1, grid_y);
    CoreRange combine_range(combine_core_logical, combine_core_logical);
    CoreRangeSet combine_cores(std::vector<CoreRange>{combine_range});

    CoreCoord return_core_logical(2, grid_y);
    CoreRange return_range(return_core_logical, return_core_logical);
    CoreRangeSet return_cores(std::vector<CoreRange>{return_range});

    // All cores (for semaphores)
    std::vector<CoreRange> all_core_ranges = {compute_range, CoreRange(CoreCoord(0, grid_y), CoreCoord(1, grid_y))};
    if (enable_fabric_return) {
        all_core_ranges.push_back(return_range);
    }
    CoreRangeSet all_cores(all_core_ranges);

    // ----- Physical coordinates -----
    std::vector<std::pair<uint32_t, uint32_t>> phys_coords;
    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = 0; x < grid_x; ++x) {
            auto phys = device->worker_core_from_logical_core(CoreCoord(x, y));
            phys_coords.push_back({phys.x, phys.y});
        }
    }
    uint32_t leader_phys_x = phys_coords[0].first;
    uint32_t leader_phys_y = phys_coords[0].second;

    auto combine_phys = device->worker_core_from_logical_core(combine_core_logical);
    auto return_phys = device->worker_core_from_logical_core(return_core_logical);

    uint32_t signal_target_phys_x = enable_fabric_return ? return_phys.x : combine_phys.x;
    uint32_t signal_target_phys_y = enable_fabric_return ? return_phys.y : combine_phys.y;

    // ----- Semaphores -----
    CreateSemaphore(program, all_cores, 0);  // 0: SEM_BARRIER
    CreateSemaphore(program, all_cores, 0);  // 1: SEM_GO
    CreateSemaphore(program, all_cores, 0);  // 2: SEM_PKT_READY
    CreateSemaphore(program, all_cores, 0);  // 3: SEM_EXPERT_DONE / SEM_EXPERT_READY
    if (enable_fabric_return) {
        CreateSemaphore(program, all_cores, 0);  // 4: SEM_RECV
    }

    // ----- Circular Buffers -----
    auto cb0_config = CircularBufferConfig(BF16_TILE_BYTES, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                          .set_page_size(tt::CBIndex::c_0, BF16_TILE_BYTES);
    CreateCircularBuffer(program, compute_cores, cb0_config);

    auto cb1_config =
        CircularBufferConfig(n_weight_per_core_gu * BFP4_TILE_BYTES, {{tt::CBIndex::c_1, tt::DataFormat::Bfp4_b}})
            .set_page_size(tt::CBIndex::c_1, BFP4_TILE_BYTES);
    CreateCircularBuffer(program, compute_cores, cb1_config);

    auto cb2_config =
        CircularBufferConfig(n_out_per_core * BF16_TILE_BYTES, {{tt::CBIndex::c_2, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_2, BF16_TILE_BYTES);
    CreateCircularBuffer(program, compute_cores, cb2_config);

    auto cb3_config = CircularBufferConfig(BF16_TILE_BYTES, {{tt::CBIndex::c_3, tt::DataFormat::Float16_b}})
                          .set_page_size(tt::CBIndex::c_3, BF16_TILE_BYTES);
    CreateCircularBuffer(program, dispatch_cores, cb3_config);

    // CB3 on compute cores: fragment buffer for Phase B writer-side untilize.
    // Each core extracts n_per_core_dn tile columns into a contiguous fragment per row.
    const uint32_t frag_bytes = n_per_core_dn * TILE_HW * 2;  // 6 * 32 * 2 = 384
    auto cb_frag_config = CircularBufferConfig(frag_bytes, {{tt::CBIndex::c_3, tt::DataFormat::Float16_b}})
                              .set_page_size(tt::CBIndex::c_3, frag_bytes);
    CreateCircularBuffer(program, compute_cores, cb_frag_config);

    // Combine core CBs: when fabric_return, used by recv_accumulate (ROW_MAJOR rows, need 3 tiles)
    const uint32_t combine_cb_size = enable_fabric_return ? 3 * BF16_TILE_BYTES : BF16_TILE_BYTES;
    auto cb4_config = CircularBufferConfig(combine_cb_size, {{tt::CBIndex::c_4, tt::DataFormat::Float16_b}})
                          .set_page_size(tt::CBIndex::c_4, combine_cb_size);
    CreateCircularBuffer(program, combine_cores, cb4_config);

    auto cb5_config = CircularBufferConfig(combine_cb_size, {{tt::CBIndex::c_5, tt::DataFormat::Float16_b}})
                          .set_page_size(tt::CBIndex::c_5, combine_cb_size);
    CreateCircularBuffer(program, combine_cores, cb5_config);

    if (enable_fabric_return) {
        // CB0 on return core: fragment page buffer for ROW_MAJOR out_bufs reads.
        // Holds P * num_cores pages of frag_bytes each (total same as old tile buffer).
        // After batch read, row r is contiguous at offset r * D_bytes.
        const uint32_t P = TILE_HW;  // 32 rows per expert
        const uint32_t total_frag_pages = P * num_cores;
        auto ret_cb0_config =
            CircularBufferConfig(total_frag_pages * frag_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_0, frag_bytes);
        CreateCircularBuffer(program, return_cores, ret_cb0_config);

        // CB3 on return core: L1 buffer for metadata read from DRAM
        if (tensor_args.return_metadata_tensor.has_value()) {
            uint32_t meta_page_size = tensor_args.return_metadata_tensor.value().buffer()->page_size();
            auto ret_cb3_config = CircularBufferConfig(meta_page_size, {{tt::CBIndex::c_3, tt::DataFormat::Float16_b}})
                                      .set_page_size(tt::CBIndex::c_3, meta_page_size);
            CreateCircularBuffer(program, return_cores, ret_cb3_config);
        }
    }

    // ----- Compile-time args -----
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(tensor_args.pkt_buf.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(tensor_args.inter_buf.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(tensor_args.gate_up_weights[0].buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(tensor_args.inter_buf.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(tensor_args.down_weights[0].buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(tensor_args.out_bufs[0].buffer()).append_to(writer_ct_args);

    std::vector<uint32_t> compute_ct_args = {
        k_tiles,
        n_weight_per_core_gu,
        n_per_core_dn,
        num_experts,
        N_BLOCK_GU,
        N_BLOCK_DN,
        k_tiles_dn,
    };

    std::vector<uint32_t> dispatch_ct_args;
    TensorAccessorArgs(tensor_args.hidden_states.buffer()).append_to(dispatch_ct_args);
    TensorAccessorArgs(tensor_args.pkt_buf.buffer()).append_to(dispatch_ct_args);

    std::vector<uint32_t> combine_ct_args;
    TensorAccessorArgs(tensor_args.output.buffer()).append_to(combine_ct_args);

    std::vector<uint32_t> return_ct_args;
    if (enable_fabric_return) {
        TT_FATAL(
            tensor_args.return_metadata_tensor.has_value(),
            "return_metadata_tensor is required when enable_fabric_return is true");
        TensorAccessorArgs(tensor_args.out_bufs[0].buffer()).append_to(return_ct_args);  // CT <0> outbuf
        TensorAccessorArgs(tensor_args.output.buffer()).append_to(return_ct_args);       // CT <1> output
        TensorAccessorArgs(tensor_args.return_metadata_tensor.value().buffer())
            .append_to(return_ct_args);  // CT <2> metadata
        if (fabric_send_enabled && tensor_args.recv_staging_buf.has_value()) {
            TensorAccessorArgs(tensor_args.recv_staging_buf.value().buffer())
                .append_to(return_ct_args);  // CT <3> staging
        }
    }

    std::vector<uint32_t> recv_ct_args;
    if (fabric_send_enabled && tensor_args.recv_staging_buf.has_value()) {
        TensorAccessorArgs(tensor_args.recv_staging_buf.value().buffer()).append_to(recv_ct_args);
        TensorAccessorArgs(tensor_args.output.buffer()).append_to(recv_ct_args);
    }

    // ----- Kernel definitions -----
    auto reader_kernel_id = CreateKernel(
        program,
        READER_KERNEL,
        compute_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args});

    auto writer_kernel_id = CreateKernel(
        program,
        WRITER_KERNEL,
        compute_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct_args});

    auto compute_kernel_id = CreateKernel(
        program,
        COMPUTE_KERNEL,
        compute_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = true,
            .math_approx_mode = true,
            .compile_args = compute_ct_args});

    auto dispatch_kernel_id = CreateKernel(
        program,
        DISPATCH_KERNEL,
        dispatch_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = dispatch_ct_args});

    auto combine_kernel_id = CreateKernel(
        program,
        COMBINE_KERNEL,
        combine_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = combine_ct_args});

    // Return kernel (with optional ENABLE_FABRIC_SEND define)
    std::optional<KernelHandle> return_kernel_id;
    std::optional<KernelHandle> recv_kernel_id;
    std::optional<CoreCoord> return_core_opt;

    if (enable_fabric_return) {
        std::map<std::string, std::string> return_defines;
        if (fabric_send_enabled) {
            return_defines["ENABLE_FABRIC_SEND"] = "1";
        }

        return_kernel_id = CreateKernel(
            program,
            RETURN_KERNEL,
            return_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = return_ct_args,
                .defines = return_defines});
        return_core_opt = return_core_logical;

        // Recv kernel on combine core (repurposed as recv accumulate for fabric return)
        // Only needed when fabric sends are active (multi-device with remote tokens).
        // Combine kernel gets num_experts=0 (no-op on dm0), recv runs on dm1.
        if (fabric_send_enabled) {
            recv_kernel_id = CreateKernel(
                program,
                RECV_KERNEL,
                CoreRangeSet(std::vector<CoreRange>{combine_range}),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = recv_ct_args});
        }
    }

    // ----- Runtime args: Reader -----
    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = 0; x < grid_x; ++x) {
            uint32_t core_idx = y * grid_x + x;
            std::vector<uint32_t> reader_args = {
                tensor_args.pkt_buf.buffer()->address(),
                k_tiles,
                tensor_args.inter_buf.buffer()->address(),
                (core_idx == 0) ? 1u : 0u,
                num_cores,
                signal_target_phys_x,
                signal_target_phys_y,
                num_experts,
                k_tiles,
            };
            for (const auto& [px, py] : phys_coords) {
                reader_args.push_back(px);
                reader_args.push_back(py);
            }
            SetRuntimeArgs(program, reader_kernel_id, CoreCoord(x, y), reader_args);
        }
    }

    // ----- Runtime args: Writer -----
    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = 0; x < grid_x; ++x) {
            uint32_t core_idx = y * grid_x + x;
            std::vector<uint32_t> writer_args = {
                tensor_args.inter_buf.buffer()->address(),  // [0]
                k_tiles,                                    // [1]
                k_tiles_dn,                                 // [2]
                n_weight_per_core_gu,                       // [3]
                n_weight_tiles_gu,                          // [4]
                core_idx * n_weight_per_core_gu,            // [5]
                core_idx * n_out_per_core,                  // [6]
                n_out_per_core,                             // [7]
                n_per_core_dn,                              // [8]
                n_tiles_dn,                                 // [9]
                core_idx * n_per_core_dn,                   // [10]
                leader_phys_x,                              // [11]
                leader_phys_y,                              // [12]
                num_experts,                                // [13]
                core_idx,                                   // [14] Phase 3: core index
                num_cores,                                  // [15] Phase 3: total cores
                TILE_HW,                                    // [16] Phase 3: P (rows per expert)
            };
            for (uint32_t e = 0; e < num_experts; ++e) {
                writer_args.push_back(tensor_args.gate_up_weights[e].buffer()->address());  // [17+e*3+0]
                writer_args.push_back(tensor_args.down_weights[e].buffer()->address());     // [17+e*3+1]
                writer_args.push_back(tensor_args.out_bufs[e].buffer()->address());         // [17+e*3+2]
            }
            SetRuntimeArgs(program, writer_kernel_id, CoreCoord(x, y), writer_args);
        }
    }

    // ----- Runtime args: Dispatch -----
    std::vector<uint32_t> dispatch_args = {
        tensor_args.hidden_states.buffer()->address(),
        tensor_args.pkt_buf.buffer()->address(),
        D_tiles,
        leader_phys_x,
        leader_phys_y,
    };
    SetRuntimeArgs(program, dispatch_kernel_id, dispatch_core_logical, dispatch_args);

    // ----- Runtime args: Combine -----
    uint32_t combine_num_experts = enable_fabric_return ? 0 : num_experts;
    std::vector<uint32_t> combine_args = {
        tensor_args.output.buffer()->address(),
        D_tiles,
        combine_num_experts,
    };
    if (!enable_fabric_return) {
        combine_args.insert(combine_args.end(), combine_metadata.begin(), combine_metadata.end());
    }
    SetRuntimeArgs(program, combine_kernel_id, combine_core_logical, combine_args);

    // ----- Runtime args: Recv kernel -----
    if (recv_kernel_id.has_value()) {
        const uint32_t D_bytes = D * 2;

        // total_expected_remote is the last element of this device's return_metadata
        uint32_t total_expected_remote = 0;
        if (device_index < attributes.return_metadata.size()) {
            const auto& dev_return_meta = attributes.return_metadata[device_index];
            if (!dev_return_meta.empty()) {
                total_expected_remote = dev_return_meta.back();
            }
        }

        // Build dest_page_ids by scanning all source devices' metadata for tokens
        // destined for this device. Per-token fields: [src_row, dest_device, dest_page,
        // recv_slot_id]. Sort by recv_slot_id to match staging buffer order.
        std::vector<std::pair<uint32_t, uint32_t>> slot_to_dest;  // (recv_slot_id, dest_page)
        for (uint32_t src_dev = 0; src_dev < attributes.return_metadata.size(); ++src_dev) {
            if (src_dev == device_index) {
                continue;
            }
            const auto& src_meta = attributes.return_metadata[src_dev];
            uint32_t idx = 0;
            for (uint32_t e = 0; e < num_experts; ++e) {
                idx++;  // skip out_buf_addr
                uint32_t M_e = src_meta[idx++];
                for (uint32_t t = 0; t < M_e; ++t) {
                    /* uint32_t src_row = */ idx++;
                    uint32_t dest_device = src_meta[idx++];
                    uint32_t dest_page = src_meta[idx++];
                    uint32_t recv_slot_id = src_meta[idx++];
                    if (dest_device == device_index) {
                        slot_to_dest.push_back({recv_slot_id, dest_page});
                    }
                }
            }
        }
        std::sort(slot_to_dest.begin(), slot_to_dest.end());

        uint32_t recv_staging_addr = 0;
        if (tensor_args.recv_staging_buf.has_value()) {
            recv_staging_addr = tensor_args.recv_staging_buf.value().buffer()->address();
        }

        std::vector<uint32_t> recv_args = {
            total_expected_remote,
            recv_staging_addr,
            tensor_args.output.buffer()->address(),
            D_bytes,
        };
        for (const auto& [slot, dest_page] : slot_to_dest) {
            recv_args.push_back(dest_page);
        }
        SetRuntimeArgs(program, recv_kernel_id.value(), combine_core_logical, recv_args);
    }

    // ----- Runtime args: Return kernel -----
    if (return_kernel_id.has_value()) {
        const uint32_t D_bytes = D * 2;
        auto recv_phys = combine_phys;

        uint32_t staging_addr_val = 0;
        if (tensor_args.recv_staging_buf.has_value()) {
            staging_addr_val = tensor_args.recv_staging_buf.value().buffer()->address();
        }

        // Compute metadata word count (return_metadata minus total_expected_remote)
        uint32_t metadata_word_count = 0;
        if (device_index < attributes.return_metadata.size()) {
            const auto& dm = attributes.return_metadata[device_index];
            metadata_word_count = dm.empty() ? 0 : static_cast<uint32_t>(dm.size() - 1);
        }

        uint32_t metadata_addr = 0;
        if (tensor_args.return_metadata_tensor.has_value()) {
            metadata_addr = tensor_args.return_metadata_tensor.value().buffer()->address();
        }

        std::vector<uint32_t> return_args = {
            tensor_args.output.buffer()->address(),  // [0]
            num_cores,                               // [1] num_cores (was D_tiles)
            D_bytes,                                 // [2]
            num_experts,                             // [3]
            device_index,                            // [4] my_device_id
            recv_phys.x,                             // [5] recv_core_phys_x
            recv_phys.y,                             // [6] recv_core_phys_y
            staging_addr_val,                        // [7] staging_addr
            metadata_addr,                           // [8] metadata DRAM address
            metadata_word_count,                     // [9] metadata word count
            TILE_HW,                                 // [10] P (rows per expert)
        };

        // Append fabric connection RT args at fixed offset [11+]
        // FabricConnectionManager::build_from_args expects:
        //   [forward_flag] [forward_connection_args if flag=1]
        //   [backward_flag] [backward_connection_args if flag=1]
        // Forward = east (col+1), backward = west (col-1)
        if (fabric_send_enabled) {
            auto src_node_id = mesh_dev->get_fabric_node_id(mesh_coordinate);
            constexpr uint32_t link = 0;
            auto mesh_shape = mesh_dev->shape();
            uint32_t coord_row = mesh_coordinate[0];
            uint32_t coord_col = mesh_coordinate[1];

            bool has_east = (coord_col + 1 < mesh_shape[1]);
            bool has_west = (coord_col > 0);

            // Forward (east) flag and connection
            return_args.push_back(has_east ? 1 : 0);
            if (has_east) {
                auto east_coord = ttnn::MeshCoordinate{coord_row, coord_col + 1};
                auto dst_node_id = mesh_dev->get_fabric_node_id(east_coord);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_node_id, dst_node_id, link, program, return_core_logical, return_args);
            }

            // Backward (west) flag and connection
            return_args.push_back(has_west ? 1 : 0);
            if (has_west) {
                auto west_coord = ttnn::MeshCoordinate{coord_row, coord_col - 1};
                auto dst_node_id = mesh_dev->get_fabric_node_id(west_coord);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_node_id, dst_node_id, link, program, return_core_logical, return_args);
            }
        }

        SetRuntimeArgs(program, return_kernel_id.value(), return_core_logical, return_args);
    }

    return {
        std::move(program),
        {reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         dispatch_kernel_id,
         combine_kernel_id,
         compute_cores,
         dispatch_core_logical,
         combine_core_logical,
         num_cores,
         grid_x,
         grid_y,
         enable_fabric_return,
         return_kernel_id,
         recv_kernel_id,
         return_core_opt}};
}

// ---------------------------------------------------------------------------
// override_runtime_arguments
// ---------------------------------------------------------------------------
void PrefillMoeComputeMeshFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output_tensor*/) {
    const uint32_t num_experts = attributes.num_experts;
    const auto& hs_shape = tensor_args.hidden_states.logical_shape();
    const uint32_t D = hs_shape[3];
    const uint32_t D_tiles = D / TILE_HW;

    uint32_t device_index = 0;
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);

        TT_FATAL(
            device_index < attributes.per_device_combine_metadata.size(), "device_index {} out of range", device_index);
        const auto& combine_metadata = attributes.per_device_combine_metadata[device_index];

        // Update reader
        for (uint32_t y = 0; y < shared.grid_y; ++y) {
            for (uint32_t x = 0; x < shared.grid_x; ++x) {
                auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id, CoreCoord(x, y));
                reader_args[0] = tensor_args.pkt_buf.buffer()->address();
                reader_args[2] = tensor_args.inter_buf.buffer()->address();
            }
        }

        // Update writer
        for (uint32_t y = 0; y < shared.grid_y; ++y) {
            for (uint32_t x = 0; x < shared.grid_x; ++x) {
                auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id, CoreCoord(x, y));
                writer_args[0] = tensor_args.inter_buf.buffer()->address();
                for (uint32_t e = 0; e < num_experts; ++e) {
                    writer_args[17 + e * 3 + 0] = tensor_args.gate_up_weights[e].buffer()->address();
                    writer_args[17 + e * 3 + 1] = tensor_args.down_weights[e].buffer()->address();
                    writer_args[17 + e * 3 + 2] = tensor_args.out_bufs[e].buffer()->address();
                }
            }
        }

        // Update dispatch
        auto& dispatch_args = GetRuntimeArgs(program, shared.dispatch_kernel_id, shared.dispatch_core);
        dispatch_args[0] = tensor_args.hidden_states.buffer()->address();
        dispatch_args[1] = tensor_args.pkt_buf.buffer()->address();

        // Rebuild combine
        uint32_t combine_num_experts = shared.enable_fabric_return ? 0 : num_experts;
        std::vector<uint32_t> combine_args = {
            tensor_args.output.buffer()->address(),
            D_tiles,
            combine_num_experts,
        };
        if (!shared.enable_fabric_return) {
            combine_args.insert(combine_args.end(), combine_metadata.begin(), combine_metadata.end());
        }
        SetRuntimeArgs(program, shared.combine_kernel_id, shared.combine_core, combine_args);

        // Rebuild return kernel args (fabric connection args are baked in, just update buffer addrs)
        if (shared.return_kernel_id.has_value() && shared.return_core.has_value()) {
            auto& return_args = GetRuntimeArgs(program, shared.return_kernel_id.value(), shared.return_core.value());
            return_args[0] = tensor_args.output.buffer()->address();
            if (tensor_args.recv_staging_buf.has_value()) {
                return_args[7] = tensor_args.recv_staging_buf.value().buffer()->address();
            }
            if (tensor_args.return_metadata_tensor.has_value()) {
                return_args[8] = tensor_args.return_metadata_tensor.value().buffer()->address();
            }
            // [9] metadata_word_count doesn't change (routing is fixed per program cache entry)
        }

        // Rebuild recv kernel args (update buffer addresses, routing metadata is fixed)
        if (shared.recv_kernel_id.has_value()) {
            auto& recv_args = GetRuntimeArgs(program, shared.recv_kernel_id.value(), shared.combine_core);
            // recv_args[0] = total_expected (doesn't change)
            if (tensor_args.recv_staging_buf.has_value()) {
                recv_args[1] = tensor_args.recv_staging_buf.value().buffer()->address();
            }
            recv_args[2] = tensor_args.output.buffer()->address();
            // recv_args[3] = D_bytes (doesn't change)
            // dest_token_ids don't change
        }

        ++device_index;
    }
}

}  // namespace ttnn::operations::experimental::prefill_moe_compute
