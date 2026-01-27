// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <unordered_set>
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

    uint32_t root_row = root_coord[0];
    uint32_t my_row = coord[0];

    // ROOT2: same row as ROOT1, different column
    if (my_row == root_row) {
        return MESH_ROOT2;
    }

    // ROOT3: the other inner row (if ROOT1 is at row 1, ROOT3 is at row 2, and vice versa)
    // Inner rows are 1 and 2 in a 4-row mesh
    uint32_t root3_row = (root_row == 1) ? 2 : 1;
    if (my_row == root3_row) {
        return MESH_ROOT3;
    }

    return MESH_LEAF;
}

inline bool is_row0_sender(const MeshCoordinate& coord) { return coord[0] == 0; }

inline bool is_row3_sender(const MeshCoordinate& coord) { return coord[0] == 3; }

// Helper struct to hold bottom core information
struct BottomCoresInfo {
    std::unordered_map<uint32_t, CoreCoord> column_to_bottom_core;
    std::unordered_set<CoreCoord> bottom_cores_lookup;
    std::vector<CoreCoord> bottom_cores_vec;
    std::vector<CoreCoord> non_bottom_cores_vec;
    std::unordered_map<CoreCoord, uint32_t> bottom_core_to_link;
};

// Build bottom core information from shard cores
// Bottom core for each column is the core with the maximum y value in that column
inline BottomCoresInfo build_bottom_cores_info(const std::vector<CoreCoord>& all_coord_cores, uint32_t num_links = 2) {
    BottomCoresInfo info;

    // Build map from column to bottom core (core with max y in each column)
    for (const auto& core : all_coord_cores) {
        auto it = info.column_to_bottom_core.find(core.x);
        if (it == info.column_to_bottom_core.end() || core.y > it->second.y) {
            info.column_to_bottom_core[core.x] = core;
        }
    }

    // Build set of bottom cores for fast lookup
    for (const auto& [col, bottom_core] : info.column_to_bottom_core) {
        info.bottom_cores_lookup.insert(bottom_core);
    }

    // Separate bottom cores from non-bottom cores
    for (const auto& core : all_coord_cores) {
        if (info.bottom_cores_lookup.count(core)) {
            info.bottom_cores_vec.push_back(core);
        } else {
            info.non_bottom_cores_vec.push_back(core);
        }
    }

    // Assign link indices to bottom cores - split evenly between links
    const uint32_t num_bottom_cores = info.bottom_cores_vec.size();
    const uint32_t cores_per_link = num_bottom_cores / num_links;
    for (uint32_t i = 0; i < num_bottom_cores; i++) {
        uint32_t link_idx = (i < cores_per_link) ? 0 : 1;
        info.bottom_core_to_link[info.bottom_cores_vec[i]] = link_idx;
    }

    return info;
}

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
    ReduceToOneOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    auto* mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor.device());
    const auto& input_tensor = tensor_args.input_tensor;
    auto* device = input_tensor.device();

    uint32_t role = get_device_role(device_coordinate, root_coord);
    bool is_mesh_leaf = (role == MESH_LEAF);
    bool is_mesh_root2 = (role == MESH_ROOT2);
    bool is_mesh_root1 = (role == MESH_ROOT1);

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

    // Build bottom core information
    auto bottom_cores_info = build_bottom_cores_info(all_coord_cores);
    const auto& column_to_bottom_core = bottom_cores_info.column_to_bottom_core;
    const auto& bottom_cores_lookup = bottom_cores_info.bottom_cores_lookup;
    const auto& bottom_cores_vec = bottom_cores_info.bottom_cores_vec;
    const auto& non_bottom_cores_vec = bottom_cores_info.non_bottom_cores_vec;
    const auto& bottom_core_to_link = bottom_cores_info.bottom_core_to_link;
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

    // === Create CBs backed by tensor buffers ===
    // num_worker_slots = non-bottom cores per column (cores that send to bottom core)
    const uint32_t num_columns = bottom_cores_vec.size();
    const uint32_t num_worker_slots = non_bottom_cores_vec.size() / num_columns;

    // Get tensors for CB backing:
    // - output_tensors[0][0] = intermediate tensor (for receiving fabric data)
    // - output_tensors[0][1] = output tensor (for compute output)
    const auto& intermediate_tensor = output_tensors[0][0];
    const auto& output_tensor = output_tensors[1][0];

    // local_cb: backed by input tensor buffer (sharded input data)
    auto cb_local_config = tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{local_cb, input_dataformat}})
                               .set_page_size(local_cb, payload_size_bytes)
                               .set_tile_dims(local_cb, compute_tile)
                               .set_globally_allocated_address(*input_tensor.buffer());
    CreateCircularBuffer(program, all_cores, cb_local_config);

    // received_cb: backed by intermediate tensor (where fabric writes received data)
    // This MUST match dst_l1_addr that senders use for fabric writes
    auto cb_received_config = tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{received_cb, input_dataformat}})
                                  .set_page_size(received_cb, payload_size_bytes)
                                  .set_tile_dims(received_cb, compute_tile)
                                  .set_globally_allocated_address(*intermediate_tensor.buffer());
    CreateCircularBuffer(program, all_cores, cb_received_config);

    // output_cb: backed by output tensor (compute output destination)
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{output_cb, input_dataformat}})
                                .set_page_size(output_cb, payload_size_bytes)
                                .set_tile_dims(output_cb, compute_tile)
                                .set_globally_allocated_address(*output_tensor.buffer());
    CreateCircularBuffer(program, all_cores, cb_output_config);

    // packet_cb: staging buffer for workers to assemble packets before bottom core forwards them
    // Workers write [header][payload] here, bottom core reads and sends via fabric
    auto cb_packet_config =
        tt::tt_metal::CircularBufferConfig(num_worker_slots * slot_size_bytes, {{packet_cb, input_dataformat}})
            .set_page_size(packet_cb, slot_size_bytes)
            .set_tile_dims(packet_cb, compute_tile);
    CreateCircularBuffer(program, all_cores, cb_packet_config);

    MeshCoordinate dest_coord =
        get_fabric_destination(role, device_coordinate, root_coord, forward_coord, backward_coord);

    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(device_coordinate);
    const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(dest_coord);

    // Calculate num_hops as Manhattan distance between source and destination mesh coordinates
    auto abs_diff = [](uint32_t a, uint32_t b) -> uint32_t { return a > b ? a - b : b - a; };
    const uint32_t num_hops =
        abs_diff(device_coordinate[0], dest_coord[0]) + abs_diff(device_coordinate[1], dest_coord[1]);

    // Destination L1 address is the intermediate tensor buffer address on the destination device
    const uint32_t dst_l1_addr = intermediate_tensor.buffer()->address();

    // === Create Kernels ===

    // Reader kernel: handles data arrival and pushes to compute
    std::vector<uint32_t> reader_ct_args = {role, local_cb, received_cb, compute_num_tiles};
    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_one/device/kernels/receiver_reader_kernel.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // Worker writer kernel: non-bottom cores assemble and send packets to bottom core
    // Final result CB depends on number of reduction stages:
    //   LEAF: local_cb (no compute), ROOT3: output_cb (1 stage), ROOT2: local_cb (2 stages)
    // Kernel early exits for ROOT1 (output is in-place, no sending needed)
    uint32_t worker_source_cb = is_mesh_leaf ? local_cb : (is_mesh_root2 ? local_cb : output_cb);
    std::vector<uint32_t> worker_writer_ct_args = {
        role, worker_source_cb, compute_num_tiles, payload_size_bytes, packet_cb};
    auto worker_writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_one/device/kernels/worker_writer_kernel.cpp",
        non_bottom_cores_set,
        tt::tt_metal::WriterDataMovementConfig(worker_writer_ct_args));

    // Fabric writer kernel: bottom cores handle fabric communication
    // Final result CB: LEAF→local_cb, ROOT3→output_cb, ROOT2→local_cb, ROOT1→output_cb
    uint32_t fabric_local_cb = is_mesh_leaf ? local_cb : (is_mesh_root2 ? local_cb : output_cb);
    std::vector<uint32_t> fabric_writer_ct_args = {
        role,
        fabric_local_cb,
        output_cb,
        compute_num_tiles,
        payload_size_bytes,
        payload_size_bytes,
        num_worker_slots,
        packet_cb};
    auto fabric_writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_one/device/kernels/fabric_writer_kernel.cpp",
        bottom_cores_set,
        tt::tt_metal::WriterDataMovementConfig(fabric_writer_ct_args));

    // Compute kernel: reduction for ROOT devices, early exits for LEAF
    std::vector<uint32_t> compute_ct_args = {local_cb, received_cb, output_cb, compute_num_tiles, role};
    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_one/device/kernels/compute_kernel.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
        });

    // === Set Runtime Args ===
    std::vector<CoreCoord> cores;

    // Create worker semaphores per column on all cores
    // Each worker increments its specific semaphore on the bottom core
    // Bottom core waits on each of the local semaphores
    // worker_sems[col][worker_idx] = semaphore address
    std::vector<std::vector<uint32_t>> worker_sems(num_columns);
    for (uint32_t col = 0; col < num_columns; col++) {
        for (uint32_t worker_idx = 0; worker_idx < num_worker_slots; worker_idx++) {
            worker_sems[col].push_back(CreateSemaphore(program, all_cores, 0));
        }
    }

    for (const auto& c : all_coord_cores) {
        auto phys_core = device->worker_core_from_logical_core(c);
        auto core_noc_x = phys_core.x;
        auto core_noc_y = phys_core.y;

        CoreCoord my_bottom_core = column_to_bottom_core.at(c.x);
        auto bottom_phys = device->worker_core_from_logical_core(my_bottom_core);

        // === Reader runtime args ===
        // Always pass all 3 semaphore addresses - kernel decides which to use based on role
        std::vector<uint32_t> reader_rt_args = {
            semaphore_round1.address(), semaphore_round2.address(), semaphore_round3.address()};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_rt_args);

        // === Writer runtime args ===
        if (bottom_cores_lookup.count(c)) {
            // Fabric writer runtime args
            std::vector<uint32_t> fabric_rt_args = {
                dst_l1_addr,
                core_noc_x,    // This core's NOC X (same position on dest device)
                core_noc_y,    // This core's NOC Y (same position on dest device)
                dst_sem_addr,  // Destination semaphore for fused atomic inc
                slot_size_bytes,
                num_hops};

            // Append worker semaphore addresses (for this column)
            for (uint32_t worker_idx = 0; worker_idx < num_worker_slots; worker_idx++) {
                fabric_rt_args.push_back(worker_sems[c.x][worker_idx]);
            }

            // Append raw fabric connection args (only for non-ROOT1, ROOT1 doesn't send)
            if (!is_mesh_root1) {
                uint32_t link_idx = bottom_core_to_link.at(c);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fabric_node_id, dst_fabric_node_id, link_idx, program, c, fabric_rt_args);
            }

            tt::tt_metal::SetRuntimeArgs(program, fabric_writer_kernel, c, fabric_rt_args);

        } else {
            // Worker writer runtime args
            // Calculate slot index: rows 0,1,2 map to slots 0,1,2
            uint32_t my_slot_idx = c.y;  // Row 0,1,2 -> slot 0,1,2

            std::vector<uint32_t> worker_rt_args = {
                bottom_phys.x,
                bottom_phys.y,
                my_slot_idx,
                worker_sems[c.x][my_slot_idx],  // This worker's semaphore on bottom core
                slot_size_bytes,
                num_hops,
                core_noc_x,  // Worker's own NOC X (same position on dest device)
                core_noc_y,  // Worker's own NOC Y (same position on dest device)
                dst_l1_addr,
                dst_sem_addr  // Destination semaphore for fused atomic inc
            };

            tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel, c, worker_rt_args);
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

        // Update reader kernel runtime args - all 3 semaphore addresses for all devices
        auto& reader_runtime_args_by_core =
            tt::tt_metal::GetRuntimeArgs(program, shared_variables.send_reader_kernel_id);

        for (const auto& core : shared_variables.cores) {
            auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
            reader_runtime_args[0] = shared_variables.semaphores[0].address();
            reader_runtime_args[1] = shared_variables.semaphores[1].address();
            reader_runtime_args[2] = shared_variables.semaphores[2].address();
        }
    }
}

}  // namespace ttnn::operations::ccl
