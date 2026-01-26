// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "reduce_to_one_op.hpp"

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::ccl {

// Device roles for 2x4 mesh with 3-level reduction tree
//
// Mesh layout (coord = [col, row]):
//   [0,0] a0  |  b0 [1,0]   row 0 - sender (sends to row 1)
//   [0,1] a1  |  b1 [1,1]   row 1 - root2 (a1) / root1 (b1)
//   [0,2] a2  |  b2 [1,2]   row 2 - root3 (receives from row 3, sends to row 1)
//   [0,3] a3  |  b3 [1,3]   row 3 - sender (sends to row 2)
//
// 3-Level Reduction Tree:
//   Level 1: Row 0 senders → Row 1 (a0→a1, b0→b1)
//            Row 3 senders → Row 2 (a3→a2, b3→b2)
//   Level 2: Root3 → Root2/Root1 (a2→a1, b2→b1)
//   Level 3: Root2 → Root1 (a1→b1, cross-column)
//
// Final result at root1 (b1) = sum of all 8 devices

enum MeshRole : uint32_t { MESH_LEAF = 0, MESH_ROOT3 = 1, MESH_ROOT2 = 2, MESH_ROOT1 = 3 };

inline uint32_t get_device_role(const MeshCoordinate& coord, const MeshCoordinate& root_coord) {
    if (coord == root_coord) {
        return MESH_ROOT1;
    }
    if (coord[1] == 1) {
        return MESH_ROOT2;
    }
    if (coord[1] == 2) {
        return MESH_ROOT3;
    }
    return MESH_LEAF;
}

inline bool is_row0_sender(const MeshCoordinate& coord) { return coord[1] == 0; }

inline bool is_row3_sender(const MeshCoordinate& coord) { return coord[1] == 3; }

inline bool is_bottom_core(const CoreCoord& core) { return core.y == 3; }

inline CoreCoord get_bottom_core_for_column(uint32_t col) { return CoreCoord(col, 3); }

inline MeshCoordinate get_fabric_destination(
    uint32_t role,
    const MeshCoordinate& device_coordinate,
    const MeshCoordinate& root_coord,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord) {
    switch (role) {
        case MESH_LEAF:
            if (is_row0_sender(device_coordinate)) {
                TT_FATAL(forward_coord.has_value(), "Row 0 sender must have forward coord");
                return forward_coord.value();
            } else {
                TT_FATAL(backward_coord.has_value(), "Row 3 sender must have backward coord");
                return backward_coord.value();
            }

        case MESH_ROOT3:
            TT_FATAL(backward_coord.has_value(), "Root3 must have backward coord");
            return backward_coord.value();

        case MESH_ROOT2: return root_coord;

        case MESH_ROOT1:
        default: return device_coordinate;
    }
}

// Get the destination semaphore address based on role
// - SENDER: increments semaphore_round1 on destination
// - ROOT3: increments semaphore_round2 on destination
// - ROOT2: increments semaphore_round3 on destination
inline uint32_t get_destination_semaphore_address(
    uint32_t role,
    const tt::tt_metal::GlobalSemaphore& semaphore_round1,
    const tt::tt_metal::GlobalSemaphore& semaphore_round2,
    const tt::tt_metal::GlobalSemaphore& semaphore_round3) {
    switch (role) {
        case MESH_LEAF: return semaphore_round1.address();
        case MESH_ROOT3: return semaphore_round2.address();
        case MESH_ROOT2: return semaphore_round3.address();
        case MESH_ROOT1:
        default: return 0;  // Root1 doesn't send
    }
}

ttnn::device_operation::CachedProgram<ReduceToOneOp::ReduceToOne::shared_variables_t> reduce_to_one_program_factory(
    const ReduceToOneOp::tensor_args_t& tensor_args,
    const ReduceToOneOp::operation_attributes_t& /*operation_attributes*/,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& device_coordinate,
    std::optional<ttnn::MeshCoordinate>& forward_coord,
    std::optional<ttnn::MeshCoordinate>& backward_coord,
    ReduceToOneOp::tensor_return_value_t& /*output_tensor*/,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    auto* mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor.device());
    const auto& input_tensor = tensor_args.input_tensor;
    auto* device = input_tensor.device();

    uint32_t role = get_device_role(device_coordinate, root_coord);
    bool is_mesh_leaf = (role == MESH_LEAF);
    bool is_mesh_root3 = (role == MESH_ROOT3);
    bool is_mesh_root2 = (role == MESH_ROOT2);
    bool is_mesh_root1 = (role == MESH_ROOT1);

    log_debug(
        tt::LogOp,
        "Device {} role: {} (leaf={}, root3={}, root2={}, root1={})",
        device_coordinate,
        role,
        is_mesh_leaf,
        is_mesh_root3,
        is_mesh_root2,
        is_mesh_root1);

    auto semaphore_round1 = semaphores[0];
    auto semaphore_round2 = semaphores[1];
    auto semaphore_round3 = semaphores[2];

    // Get destination semaphore address for fused atomic inc
    uint32_t dst_sem_addr =
        get_destination_semaphore_address(role, semaphore_round1, semaphore_round2, semaphore_round3);

    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;
    const auto& shard_shape = shard_spec.shape;
    const uint32_t shard_height = shard_shape[0];
    const uint32_t shard_width = shard_shape[1];

    std::vector<CoreCoord> all_coord_cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto cores = corerange_to_cores(core_range, std::nullopt);
        all_coord_cores.insert(all_coord_cores.end(), cores.begin(), cores.end());
    }
    const CoreRangeSet all_cores = shard_grid;
    const uint32_t num_shard_cores = all_coord_cores.size();

    std::vector<CoreCoord> bottom_cores_vec, non_bottom_cores_vec;
    for (const auto& core : all_coord_cores) {
        if (is_bottom_core(core)) {
            bottom_cores_vec.push_back(core);
        } else {
            non_bottom_cores_vec.push_back(core);
        }
    }
    CoreRangeSet bottom_cores_set = CoreRangeSet(bottom_cores_vec);
    CoreRangeSet non_bottom_cores_set = CoreRangeSet(non_bottom_cores_vec);

    uint32_t input_total_num_pages = data_movement::get_num_pages(input_tensor);
    const uint32_t input_num_pages = input_total_num_pages / num_shard_cores;
    const uint32_t input_num_tiles = input_num_pages;

    const uint32_t input_page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t payload_size_bytes = input_num_tiles * input_page_size_bytes;

    // Slot size = header + payload (for worker packets sent to bottom core)
    const uint32_t packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    const uint32_t slot_size_bytes = packet_header_size_bytes + payload_size_bytes;

    tt::tt_metal::Program program{};

    // Use 32x32 tile for compute optimization
    // Calculate how many 32x32 tiles needed based on shard shape
    constexpr uint32_t compute_tile_height = 32;
    constexpr uint32_t compute_tile_width = 32;
    constexpr uint32_t compute_tile_elements = compute_tile_height * compute_tile_width;  // 1024
    const uint32_t shard_elements = shard_height * shard_width;
    const uint32_t compute_num_tiles = tt::div_up(shard_elements, compute_tile_elements);
    const auto compute_tile = tt::tt_metal::Tile({compute_tile_height, compute_tile_width});
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    constexpr auto local_cb = tt::CBIndex::c_0;
    constexpr auto received_cb = tt::CBIndex::c_1;
    constexpr auto output_cb = tt::CBIndex::c_2;
    constexpr auto packet_cb = tt::CBIndex::c_3;

    // === Create CBs on all cores for simplicity ===
    const uint32_t num_worker_slots = 3;

    auto cb_local_config = tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{local_cb, input_dataformat}})
                               .set_page_size(local_cb, payload_size_bytes)
                               .set_tile_dims(local_cb, compute_tile);
    CreateCircularBuffer(program, all_cores, cb_local_config);

    auto cb_received_config = tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{received_cb, input_dataformat}})
                                  .set_page_size(received_cb, payload_size_bytes)
                                  .set_tile_dims(received_cb, compute_tile);
    CreateCircularBuffer(program, all_cores, cb_received_config);

    auto cb_output_config = tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{output_cb, input_dataformat}})
                                .set_page_size(output_cb, payload_size_bytes)
                                .set_tile_dims(output_cb, compute_tile);
    CreateCircularBuffer(program, all_cores, cb_output_config);

    auto cb_packet_config =
        tt::tt_metal::CircularBufferConfig(num_worker_slots * slot_size_bytes, {{packet_cb, input_dataformat}})
            .set_page_size(packet_cb, slot_size_bytes)
            .set_tile_dims(packet_cb, compute_tile);
    CreateCircularBuffer(program, all_cores, cb_packet_config);

    MeshCoordinate dest_coord =
        get_fabric_destination(role, device_coordinate, root_coord, forward_coord, backward_coord);

    log_debug(tt::LogOp, "Device {} sending to {}", device_coordinate, dest_coord);

    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(device_coordinate);
    const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(dest_coord);

    // Calculate num_hops (simplified: assume 1 hop for adjacent devices)
    // TODO: Calculate actual hops based on mesh topology
    const uint32_t num_hops = 1;

    // Destination L1 address will be the received_cb address on the destination device
    // Since CB layout is deterministic, we use a placeholder that will be filled at runtime
    // For now, use 0 as placeholder - this should be fixed properly
    uint32_t dst_l1_addr = 0;  // TODO: Get actual received_cb address

    // === Create Kernels ===

    // Reader kernel: handles data arrival and pushes to compute
    std::vector<uint32_t> reader_ct_args = {role, local_cb, received_cb, compute_num_tiles};
    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_one/device/kernels/receiver_reader_kernel.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // Worker writer kernel: non-bottom cores assemble and send packets to bottom core
    // NOT created for ROOT1 (output is in-place, no sending needed)
    // Final result CB depends on number of reduction stages:
    //   LEAF: local_cb (no compute), ROOT3: output_cb (1 stage), ROOT2: local_cb (2 stages)
    tt::tt_metal::KernelHandle worker_writer_kernel = 0;
    if (!is_mesh_root1) {
        uint32_t worker_source_cb = is_mesh_leaf ? local_cb : (is_mesh_root2 ? local_cb : output_cb);
        std::vector<uint32_t> worker_writer_ct_args = {worker_source_cb, compute_num_tiles, payload_size_bytes};
        worker_writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_one/device/kernels/worker_writer_kernel.cpp",
            non_bottom_cores_set,
            tt::tt_metal::WriterDataMovementConfig(worker_writer_ct_args));
    }

    // Fabric writer kernel: bottom cores handle fabric communication
    // Final result CB: LEAF→local_cb, ROOT3→output_cb, ROOT2→local_cb, ROOT1→output_cb
    uint32_t fabric_local_cb = is_mesh_leaf ? local_cb : (is_mesh_root2 ? local_cb : output_cb);
    std::vector<uint32_t> fabric_writer_ct_args = {
        role, fabric_local_cb, output_cb, compute_num_tiles, payload_size_bytes, payload_size_bytes, num_worker_slots};
    auto fabric_writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_one/device/kernels/fabric_writer_kernel.cpp",
        bottom_cores_set,
        tt::tt_metal::WriterDataMovementConfig(fabric_writer_ct_args));

    // Compute kernel: reduction (only for ROOT devices)
    tt::tt_metal::KernelHandle compute_kernel = 0;
    if (!is_mesh_leaf) {
        std::vector<uint32_t> compute_ct_args = {local_cb, received_cb, output_cb, compute_num_tiles, role};
        compute_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_one/device/kernels/compute_kernel.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .math_approx_mode = false,
                .compile_args = compute_ct_args,
            });
    }

    // === Set Runtime Args ===
    std::vector<CoreCoord> cores;

    // Create arrival semaphores for each bottom core
    // Needed by all devices that forward worker packets (SENDER, ROOT3, ROOT2)
    std::unordered_map<CoreCoord, uint32_t> arrival_sem_map;
    if (!is_mesh_root1) {
        for (const auto& bc : bottom_cores_vec) {
            arrival_sem_map[bc] = CreateSemaphore(program, {bc}, 0);
        }
    }

    for (const auto& c : all_coord_cores) {
        auto phys_core = device->worker_core_from_logical_core(c);
        auto core_noc_x = phys_core.x;
        auto core_noc_y = phys_core.y;

        CoreCoord my_bottom_core = get_bottom_core_for_column(c.x);
        auto bottom_phys = device->worker_core_from_logical_core(my_bottom_core);

        // === Reader runtime args ===
        std::vector<uint32_t> reader_rt_args;
        if (is_mesh_leaf) {
            // Senders: no runtime args needed
            reader_rt_args = {};
        } else {
            // ROOT devices: need semaphore addresses (each round receives 1 shard)
            reader_rt_args = {semaphore_round1.address()};
            if (is_mesh_root2 || is_mesh_root1) {
                reader_rt_args.push_back(semaphore_round2.address());
            }
            if (is_mesh_root1) {
                reader_rt_args.push_back(semaphore_round3.address());
            }
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_rt_args);

        // === Writer runtime args ===
        if (is_bottom_core(c)) {
            // Fabric writer runtime args
            std::vector<uint32_t> fabric_rt_args = {
                dst_l1_addr,
                core_noc_x,                               // This core's NOC X (same position on dest device)
                core_noc_y,                               // This core's NOC Y (same position on dest device)
                dst_sem_addr,                             // Destination semaphore for fused atomic inc
                is_mesh_root1 ? 0u : arrival_sem_map[c],  // worker_arrival_sem (for all non-root1)
                0,                                        // packet_buffer_addr - placeholder
                slot_size_bytes,
                num_hops};

            // Append raw fabric connection args (always, even for root1)
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id,
                dst_fabric_node_id,
                0,  // link_idx
                program,
                c,
                fabric_rt_args);

            tt::tt_metal::SetRuntimeArgs(program, fabric_writer_kernel, c, fabric_rt_args);

        } else {
            // Worker writer runtime args (only for non-ROOT1)
            if (!is_mesh_root1) {
                // Calculate slot index: rows 0,1,2 map to slots 0,1,2
                uint32_t my_slot_idx = c.y;  // Row 0,1,2 -> slot 0,1,2

                std::vector<uint32_t> worker_rt_args = {
                    bottom_phys.x,
                    bottom_phys.y,
                    my_slot_idx,
                    0,  // packet_buffer_addr - placeholder
                    arrival_sem_map[my_bottom_core],
                    slot_size_bytes,
                    num_hops,
                    core_noc_x,  // Worker's own NOC X (same position on dest device)
                    core_noc_y,  // Worker's own NOC Y (same position on dest device)
                    dst_l1_addr,
                    dst_sem_addr  // Destination semaphore for fused atomic inc
                };

                tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel, c, worker_rt_args);
            }
        }

        cores.push_back(c);
    }

    return {
        std::move(program),
        ReduceToOneOp::ReduceToOne::shared_variables_t{
            .send_reader_kernel_id = reader_kernel,
            .send_writer_kernel_id = worker_writer_kernel,
            .cores = cores,
            .root1_reader_kernel_id = is_mesh_root1 ? reader_kernel : 0,
            .root1_writer_kernel_id = is_mesh_root1 ? fabric_writer_kernel : 0,
            .root2_reader_kernel_id = is_mesh_root2 ? reader_kernel : 0,
            .root2_writer_kernel_id = is_mesh_root2 ? fabric_writer_kernel : 0,
            .compute_kernel_id = compute_kernel,
            .semaphores = semaphores,
            .is_mesh_leaf_device = is_mesh_leaf,
            .is_root_device = is_mesh_root1,
            .is_mesh_root2_device = is_mesh_root2,
            .is_col_root_device = false}};
}

void ReduceToOneOp::ReduceToOne::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        bool is_mesh_leaf = shared_variables.is_mesh_leaf_device;
        bool is_mesh_root1 = shared_variables.is_root_device;
        bool is_mesh_root2 = shared_variables.is_mesh_root2_device;

        // Update reader kernel runtime args (semaphore addresses for ROOT devices)
        if (!is_mesh_leaf) {
            auto& reader_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.send_reader_kernel_id);

            for (const auto& core : shared_variables.cores) {
                auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
                // Update semaphore addresses
                reader_runtime_args[0] = shared_variables.semaphores[0].address();  // recv_sem_round1
                if (is_mesh_root2 || is_mesh_root1) {
                    reader_runtime_args[1] = shared_variables.semaphores[1].address();  // recv_sem_round2
                }
                if (is_mesh_root1) {
                    reader_runtime_args[2] = shared_variables.semaphores[2].address();  // recv_sem_round3
                }
            }
        }

        // Update writer kernel runtime args
        // For fabric writer (bottom cores) and worker writer (non-bottom cores)
        // The semaphore addresses and destination addresses may need updating
        // Note: Worker writer uses local semaphores which don't change per-invocation
        (void)shared_variables.send_writer_kernel_id;  // Suppress unused warning

        if (is_mesh_root1 && shared_variables.root1_writer_kernel_id != 0) {
            auto& writer_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.root1_writer_kernel_id);

            for (const auto& core : shared_variables.cores) {
                auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
                // Fabric writer: dst_sem_addr at index 3
                writer_runtime_args[3] = shared_variables.semaphores[2].address();  // Round 3 semaphore
            }
        }

        if (is_mesh_root2 && shared_variables.root2_writer_kernel_id != 0) {
            auto& writer_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.root2_writer_kernel_id);

            for (const auto& core : shared_variables.cores) {
                auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
                // Fabric writer: dst_sem_addr at index 3
                writer_runtime_args[3] = shared_variables.semaphores[1].address();  // Round 2 semaphore
            }
        }
    }
}

}  // namespace ttnn::operations::ccl
