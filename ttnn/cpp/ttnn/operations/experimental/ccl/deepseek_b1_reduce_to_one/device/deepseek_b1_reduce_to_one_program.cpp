// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <unordered_set>
#include "deepseek_b1_reduce_to_one_op.hpp"

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::ccl {

// Device roles for 4x2 mesh (4 rows by 2 columns) with 3-level reduction tree
//
// Mesh layout (coord = [row, col]):
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

// Helper struct to hold core layout information
struct CoresInfo {
    std::unordered_map<uint32_t, CoreCoord> column_to_bottom_core;   // x_coord -> bottom core (max y in column)
    std::unordered_map<uint32_t, uint32_t> column_x_to_idx;          // x_coord -> column index (0, 1, ...)
    std::unordered_map<CoreCoord, uint32_t> core_to_slot_idx;        // worker core -> slot index within column
    std::vector<CoreCoord> fabric_writer_cores;                      // dedicated fabric writer cores (one per link)
    std::unordered_map<CoreCoord, uint32_t> fabric_core_to_link;     // fabric core -> link index
    std::unordered_map<CoreCoord, uint32_t> fabric_core_to_col_idx;  // fabric core -> column index
    std::unordered_map<uint32_t, CoreCoord> column_to_fabric_core;   // column x_coord -> fabric core for that column
};

// Build core layout information from shard cores
// All shard cores are workers, dedicated fabric cores are placed adjacent to bottom cores
inline CoresInfo build_cores_info(const std::vector<CoreCoord>& all_coord_cores, uint32_t num_links = 2) {
    CoresInfo info;

    // Build map from column (x value) to bottom core (core with max y in each column)
    for (const auto& core : all_coord_cores) {
        auto it = info.column_to_bottom_core.find(core.x);
        if (it == info.column_to_bottom_core.end() || core.y > it->second.y) {
            info.column_to_bottom_core[core.x] = core;
        }
    }

    // Map each x coordinate to a column index (0, 1, 2, ...)
    std::vector<uint32_t> sorted_x_coords;
    sorted_x_coords.reserve(info.column_to_bottom_core.size());
    for (const auto& [x_coord, _] : info.column_to_bottom_core) {
        sorted_x_coords.push_back(x_coord);
    }
    std::sort(sorted_x_coords.begin(), sorted_x_coords.end());
    for (uint32_t col_idx = 0; col_idx < sorted_x_coords.size(); col_idx++) {
        info.column_x_to_idx[sorted_x_coords[col_idx]] = col_idx;
    }

    // All shard cores are workers - assign slot indices within each column
    std::unordered_map<uint32_t, std::vector<CoreCoord>> column_workers;
    for (const auto& core : all_coord_cores) {
        column_workers[core.x].push_back(core);
    }
    for (auto& [x_coord, workers] : column_workers) {
        std::sort(workers.begin(), workers.end(), [](const CoreCoord& a, const CoreCoord& b) { return a.y < b.y; });
        for (uint32_t slot = 0; slot < workers.size(); slot++) {
            info.core_to_slot_idx[workers[slot]] = slot;
        }
    }

    // Dedicated fabric writer cores: placed at x = bottom_core.x + 1 (to the right of bottom cores)
    // Split columns between links (first half -> link 0, second half -> link 1)
    const uint32_t num_columns = sorted_x_coords.size();
    const uint32_t cols_per_link = num_columns / num_links;
    for (uint32_t col_idx = 0; col_idx < num_columns; col_idx++) {
        uint32_t x_coord = sorted_x_coords[col_idx];
        const auto& bottom_core = info.column_to_bottom_core[x_coord];
        CoreCoord fabric_core = {bottom_core.x + 1, bottom_core.y};  // To the right of bottom core
        info.fabric_writer_cores.push_back(fabric_core);
        info.column_to_fabric_core[x_coord] = fabric_core;
        info.fabric_core_to_col_idx[fabric_core] = col_idx;  // Map fabric core -> column index
        uint32_t link_idx = (col_idx < cols_per_link) ? 0 : 1;
        info.fabric_core_to_link[fabric_core] = link_idx;
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
// - LEAF: increments semaphore_round1 on destination
// - ROOT3: increments semaphore_round2 on destination
// - ROOT2: increments semaphore_round3 on destination
// - ROOT1: increments semaphore_exit on exit_coord
inline uint32_t get_destination_semaphore_address(
    uint32_t role,
    const tt::tt_metal::GlobalSemaphore& semaphore_round1,
    const tt::tt_metal::GlobalSemaphore& semaphore_round2,
    const tt::tt_metal::GlobalSemaphore& semaphore_round3,
    const tt::tt_metal::GlobalSemaphore& semaphore_exit) {
    switch (role) {
        case MESH_LEAF: return semaphore_round1.address();
        case MESH_ROOT3: return semaphore_round2.address();
        case MESH_ROOT2: return semaphore_round3.address();
        case MESH_ROOT1: return semaphore_exit.address();
        default: return 0;
    }
}

ttnn::device_operation::CachedProgram<DeepseekB1ReduceToOneOp::DeepseekB1ReduceToOne::shared_variables_t>
deepseek_b1_reduce_to_one_program_factory(
    const DeepseekB1ReduceToOneOp::tensor_args_t& tensor_args,
    const DeepseekB1ReduceToOneOp::operation_attributes_t& /*operation_attributes*/,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& exit_coord,
    const MeshCoordinate& device_coordinate,
    std::optional<ttnn::MeshCoordinate>& forward_coord,
    std::optional<ttnn::MeshCoordinate>& backward_coord,
    DeepseekB1ReduceToOneOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    auto* mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor.device());
    const auto& input_tensor = tensor_args.input_tensor;
    auto* device = input_tensor.device();

    uint32_t role = get_device_role(device_coordinate, root_coord);
    bool is_mesh_leaf = (role == MESH_LEAF);
    bool is_mesh_root3 = (role == MESH_ROOT3);
    bool is_mesh_root2 = (role == MESH_ROOT2);
    bool is_mesh_root1 = (role == MESH_ROOT1);

    auto semaphore_round1 = semaphores[0];
    auto semaphore_round2 = semaphores[1];
    auto semaphore_round3 = semaphores[2];
    auto semaphore_exit = semaphores[3];

    // Get destination semaphore address for fused atomic inc
    uint32_t dst_sem_addr =
        get_destination_semaphore_address(role, semaphore_round1, semaphore_round2, semaphore_round3, semaphore_exit);

    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;
    const auto& shard_shape = shard_spec.shape;
    const uint32_t shard_height = shard_shape[0];
    const uint32_t shard_width = shard_shape[1];

    // Get cores in shard order (respecting shard orientation)
    // This determines each core's shard index for output gather
    std::vector<CoreCoord> all_coord_cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto cores =
            corerange_to_cores(core_range, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
        all_coord_cores.insert(all_coord_cores.end(), cores.begin(), cores.end());
    }
    const CoreRangeSet all_cores = shard_grid;
    const uint32_t num_shard_cores = all_coord_cores.size();

    // Build core to shard index mapping (based on shard order)
    std::unordered_map<CoreCoord, uint32_t> core_to_shard_idx;
    for (uint32_t i = 0; i < all_coord_cores.size(); i++) {
        core_to_shard_idx[all_coord_cores[i]] = i;
    }

    // Build core layout information
    auto cores_info = build_cores_info(all_coord_cores);
    const auto& core_to_slot_idx = cores_info.core_to_slot_idx;
    const auto& fabric_writer_cores = cores_info.fabric_writer_cores;
    const auto& fabric_core_to_link = cores_info.fabric_core_to_link;
    const auto& column_to_fabric_core = cores_info.column_to_fabric_core;
    CoreRangeSet fabric_cores_set = CoreRangeSet(fabric_writer_cores);

    // Combined core set: all shard cores + fabric cores (for CB creation)
    std::vector<CoreCoord> all_cores_plus_fabric = all_coord_cores;
    all_cores_plus_fabric.insert(all_cores_plus_fabric.end(), fabric_writer_cores.begin(), fabric_writer_cores.end());
    CoreRangeSet all_cores_with_fabric = CoreRangeSet(all_cores_plus_fabric);

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
    TT_FATAL(compute_num_tiles == 1, "compute_num_tiles must be 1");
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // CB size must be tile-aligned for compute kernel
    const uint32_t tile_size_bytes = compute_tile_height * compute_tile_width * tt::datum_size(input_dataformat);
    const uint32_t cb_size_bytes = compute_num_tiles * tile_size_bytes;

    constexpr auto local_cb = tt::CBIndex::c_0;        // Input tensor (reader pushes)
    constexpr auto received_cb_r1 = tt::CBIndex::c_1;  // Round 1: LEAF → ROOT* (backed by intermediate tensor)
    constexpr auto received_cb_r2 = tt::CBIndex::c_5;  // Round 2: ROOT3 → ROOT2/ROOT1 (separate L1 buffer)
    constexpr auto received_cb_r3 = tt::CBIndex::c_6;  // Round 3: ROOT2 → ROOT1 (separate L1 buffer)
    constexpr auto output_cb = tt::CBIndex::c_2;       // Final output (backed by output tensor)
    constexpr auto packet_cb = tt::CBIndex::c_3;
    constexpr auto scratch_cb = tt::CBIndex::c_7;  // Scratch for compute (NOT globally allocated - stable addr)

    // === Create CBs backed by tensor buffers ===
    // num_worker_slots = all shard cores per column (all cores send to fabric core)
    const uint32_t num_columns = fabric_writer_cores.size();
    const uint32_t num_worker_slots = all_coord_cores.size() / num_columns;

    // Get tensors for CB backing:
    // - output_tensors[0][0] = intermediate tensor for round 1 (LEAF → ROOT*)
    // - output_tensors[0][1] = intermediate tensor for round 2 (ROOT3 → ROOT2/ROOT1)
    // - output_tensors[0][2] = intermediate tensor for round 3 (ROOT2 → ROOT1)
    // - output_tensors[1][0] = output tensor (for compute output)
    const auto& intermediate_tensor_r1 = output_tensors[0][0];
    const auto& intermediate_tensor_r2 = output_tensors[0][1];
    const auto& intermediate_tensor_r3 = output_tensors[0][2];
    const auto& output_tensor = output_tensors[1][0];

    // Get output core from output tensor's shard spec (for ROOT1 gather)
    // Validation ensures output is sharded on exactly 1 core
    const auto& output_shard_spec = output_tensor.shard_spec().value();
    const CoreCoord output_core = output_shard_spec.grid.ranges().begin()->start_coord;

    // Create all CBs on all_cores_with_fabric (shard cores + fabric cores)

    // local_cb: backed by input tensor buffer (sharded input data)
    // Must use payload_size_bytes since it's backed by tensor buffer
    auto cb_local_config = tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{local_cb, input_dataformat}})
                               .set_page_size(local_cb, payload_size_bytes)
                               .set_globally_allocated_address(*input_tensor.buffer());
    auto local_cb_handle = CreateCircularBuffer(program, all_cores_with_fabric, cb_local_config);

    // received_cb_r1: Round 1 receive buffer - backed by intermediate tensor r1
    auto cb_received_r1_config =
        tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{received_cb_r1, input_dataformat}})
            .set_page_size(received_cb_r1, payload_size_bytes)
            .set_tile_dims(received_cb_r1, compute_tile)
            .set_globally_allocated_address(*intermediate_tensor_r1.buffer());
    auto received_cb_r1_handle = CreateCircularBuffer(program, all_cores_with_fabric, cb_received_r1_config);

    // received_cb_r2: Round 2 receive buffer - backed by intermediate tensor r2
    auto cb_received_r2_config =
        tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{received_cb_r2, input_dataformat}})
            .set_page_size(received_cb_r2, payload_size_bytes)
            .set_tile_dims(received_cb_r2, compute_tile)
            .set_globally_allocated_address(*intermediate_tensor_r2.buffer());
    auto received_cb_r2_handle = CreateCircularBuffer(program, all_cores_with_fabric, cb_received_r2_config);

    // received_cb_r3: Round 3 receive buffer - backed by intermediate tensor r3
    auto cb_received_r3_config =
        tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{received_cb_r3, input_dataformat}})
            .set_page_size(received_cb_r3, payload_size_bytes)
            .set_tile_dims(received_cb_r3, compute_tile)
            .set_globally_allocated_address(*intermediate_tensor_r3.buffer());
    auto received_cb_r3_handle = CreateCircularBuffer(program, all_cores_with_fabric, cb_received_r3_config);

    // output_cb: backed by output tensor (compute output destination)
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(payload_size_bytes, {{output_cb, input_dataformat}})
                                .set_page_size(output_cb, payload_size_bytes)
                                .set_globally_allocated_address(*output_tensor.buffer());
    auto output_cb_handle = CreateCircularBuffer(program, all_cores_with_fabric, cb_output_config);

    // scratch_cb: scratch buffer for compute (NOT globally allocated - stable address)
    auto cb_scratch_config = tt::tt_metal::CircularBufferConfig(cb_size_bytes, {{scratch_cb, input_dataformat}})
                                 .set_page_size(scratch_cb, tile_size_bytes)
                                 .set_tile_dims(scratch_cb, compute_tile);
    CreateCircularBuffer(program, all_cores_with_fabric, cb_scratch_config);

    // packet_cb: staging buffer for workers to assemble packets before fabric core forwards them
    auto cb_packet_config =
        tt::tt_metal::CircularBufferConfig(num_worker_slots * slot_size_bytes, {{packet_cb, input_dataformat}})
            .set_page_size(packet_cb, slot_size_bytes);
    CreateCircularBuffer(program, all_cores_with_fabric, cb_packet_config);

    // For ROOT1, use exit_coord as destination; otherwise use normal fabric destination
    MeshCoordinate dest_coord =
        is_mesh_root1 ? exit_coord
                      : get_fabric_destination(role, device_coordinate, root_coord, forward_coord, backward_coord);

    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(device_coordinate);
    const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(dest_coord);

    // Log fabric node mapping for debugging
    const char* role_names[] = {"LEAF", "ROOT3", "ROOT2", "ROOT1"};
    log_debug(
        tt::LogOp,
        "ReduceToOne: device[{},{}] role={} src_fabric_node={} -> dest[{},{}] dst_fabric_node={}",
        device_coordinate[0],
        device_coordinate[1],
        role_names[role],
        src_fabric_node_id,
        dest_coord[0],
        dest_coord[1],
        dst_fabric_node_id);

    // Set hop = 1 for neighbour exchange
    const uint32_t num_hops = 1;

    // Destination L1 address depends on role - each round writes to a different buffer
    // - LEAF: writes to received_cb_r1 (intermediate_tensor_r1)
    // - ROOT3: writes to received_cb_r2 (intermediate_tensor_r2)
    // - ROOT2: writes to received_cb_r3 (intermediate_tensor_r3)
    uint32_t dst_l1_addr;
    switch (role) {
        case MESH_LEAF: dst_l1_addr = intermediate_tensor_r1.buffer()->address(); break;
        case MESH_ROOT3: dst_l1_addr = intermediate_tensor_r2.buffer()->address(); break;
        case MESH_ROOT2: dst_l1_addr = intermediate_tensor_r3.buffer()->address(); break;
        default: dst_l1_addr = intermediate_tensor_r1.buffer()->address(); break;  // ROOT1 doesn't send
    }

    // === Create Kernels ===

    // Reader kernel: handles data arrival and pushes to compute
    // Args: [0] role, [1] num_tiles, [2-5] CBs (local, received r1/r2/r3)
    std::vector<uint32_t> reader_ct_args = {
        role, compute_num_tiles, local_cb, received_cb_r1, received_cb_r2, received_cb_r3};
    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_b1_reduce_to_one/device/kernels/"
        "deepseek_b1_receiver_reader_kernel.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // Worker writer kernel: ALL shard cores assemble and send packets to dedicated fabric core
    // LEAF: reads from local_cb (no compute), others: reads from scratch_cb (compute output)
    // ROOT1: gathers final results to output core
    // Args: [0] role, [1-2] sizes, [3-4] CBs, [5-7] routing, [8-9] output coords
    uint32_t worker_source_cb = is_mesh_leaf ? local_cb : scratch_cb;
    auto output_phys_core = device->worker_core_from_logical_core(output_core);
    std::vector<uint32_t> worker_writer_ct_args = {
        role,
        compute_num_tiles,
        payload_size_bytes,
        worker_source_cb,
        packet_cb,
        num_hops,
        dst_fabric_node_id.chip_id,
        dst_fabric_node_id.mesh_id.get(),
        output_phys_core.x,
        output_phys_core.y};
    auto worker_writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_b1_reduce_to_one/device/kernels/"
        "deepseek_b1_worker_writer_kernel.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(worker_writer_ct_args));

    // Fabric writer kernel: dedicated fabric cores handle fabric communication
    // These cores only forward worker packets, they don't have their own shard data
    // Args: [0] role, [1-2] sizes, [3] CB
    std::vector<uint32_t> fabric_writer_ct_args = {role, num_worker_slots, payload_size_bytes, packet_cb};
    auto fabric_writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_b1_reduce_to_one/device/kernels/"
        "deepseek_b1_fabric_writer_kernel.cpp",
        fabric_cores_set,
        tt::tt_metal::WriterDataMovementConfig(fabric_writer_ct_args));

    // Compute kernel: reduction for ROOT devices, early exits for LEAF
    // Args: [0] role, [1] num_tiles, [2-7] CBs (local, received r1/r2/r3, output, scratch)
    std::vector<uint32_t> compute_ct_args = {
        role, compute_num_tiles, local_cb, received_cb_r1, received_cb_r2, received_cb_r3, output_cb, scratch_cb};
    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_b1_reduce_to_one/device/kernels/"
        "deepseek_b1_compute_kernel.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
        });

    // === Set Runtime Args ===
    std::vector<CoreCoord> cores;

    // Create worker semaphores on fabric cores
    // Each worker increments its specific semaphore on the fabric core
    // Fabric core waits on each of the local semaphores
    // worker_sems[worker_idx] = semaphore ID (same ID used on all fabric cores)
    std::vector<uint32_t> worker_sems;
    worker_sems.reserve(num_worker_slots);
    for (uint32_t worker_idx = 0; worker_idx < num_worker_slots; worker_idx++) {
        worker_sems.push_back(CreateSemaphore(program, fabric_cores_set, 0));
    }

    uint32_t output_base_addr = output_tensor.buffer()->address();

    // Set runtime args for all shard cores (workers)
    for (const auto& c : all_coord_cores) {
        // Get this core's column index and fabric core
        CoreCoord my_fabric_core = column_to_fabric_core.at(c.x);
        auto fabric_phys = device->worker_core_from_logical_core(my_fabric_core);

        // === Reader runtime args ===
        std::vector<uint32_t> reader_rt_args = {
            semaphore_round1.address(), semaphore_round2.address(), semaphore_round3.address()};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_rt_args);

        uint32_t my_slot_idx = core_to_slot_idx.at(c);

        // === Worker writer runtime args ===
        // Get this core's shard index (position in shard order for output gather)
        uint32_t shard_idx = core_to_shard_idx.at(c);

        std::vector<uint32_t> worker_rt_args = {
            fabric_phys.x,             // [0] Fabric core NOC X (per-column)
            fabric_phys.y,             // [1] Fabric core NOC Y (per-column)
            my_slot_idx,               // [2] per-core
            worker_sems[my_slot_idx],  // [3] per-core semaphore
            dst_l1_addr,               // [4] updated in trace
            dst_sem_addr,              // [5] updated in trace
            output_base_addr,          // [6] updated in trace
            shard_idx                  // [7] per-core
        };
        tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel, c, worker_rt_args);

        cores.push_back(c);
    }

    // Set runtime args for dedicated fabric cores
    for (const auto& fc : fabric_writer_cores) {
        std::vector<uint32_t> fabric_rt_args;
        fabric_rt_args.reserve(num_worker_slots);

        // Append worker semaphore IDs (fabric kernel will convert to addresses)
        for (uint32_t worker_idx = 0; worker_idx < num_worker_slots; worker_idx++) {
            fabric_rt_args.push_back(worker_sems[worker_idx]);
        }

        // Append fabric connection args
        uint32_t link_idx = fabric_core_to_link.at(fc);
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fabric_node_id, dst_fabric_node_id, link_idx, program, fc, fabric_rt_args);

        tt::tt_metal::SetRuntimeArgs(program, fabric_writer_kernel, fc, fabric_rt_args);
    }

    return {
        std::move(program),
        DeepseekB1ReduceToOneOp::DeepseekB1ReduceToOne::shared_variables_t{
            .send_reader_kernel_id = reader_kernel,
            .send_worker_writer_kernel_id = worker_writer_kernel,
            .send_fabric_writer_kernel_id = fabric_writer_kernel,
            .cores = cores,
            .fabric_cores = fabric_writer_cores,
            .root1_reader_kernel_id = is_mesh_root1 ? reader_kernel : 0,
            .root1_writer_kernel_id = is_mesh_root1 ? fabric_writer_kernel : 0,
            .root2_reader_kernel_id = is_mesh_root2 ? reader_kernel : 0,
            .root2_writer_kernel_id = is_mesh_root2 ? fabric_writer_kernel : 0,
            .compute_kernel_id = compute_kernel,
            .semaphores = semaphores,
            .is_mesh_leaf_device = is_mesh_leaf,
            .is_mesh_root3_device = is_mesh_root3,
            .is_mesh_root2_device = is_mesh_root2,
            .is_mesh_root1_device = is_mesh_root1,
            .local_cb_handle = local_cb_handle,
            .received_cb_r1_handle = received_cb_r1_handle,
            .received_cb_r2_handle = received_cb_r2_handle,
            .received_cb_r3_handle = received_cb_r3_handle,
            .output_cb_handle = output_cb_handle}};
}

void DeepseekB1ReduceToOneOp::DeepseekB1ReduceToOne::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        const auto& input_tensor = tensor_args.input_tensor;
        const auto& intermediate_tensors = tensor_return_value[0];
        const auto& output_tensor = tensor_return_value[1][0];

        // Update CB addresses for tensor buffers
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.local_cb_handle, *input_tensor.buffer());
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.received_cb_r1_handle, *intermediate_tensors[0].buffer());
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.received_cb_r2_handle, *intermediate_tensors[1].buffer());
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.received_cb_r3_handle, *intermediate_tensors[2].buffer());
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.output_cb_handle, *output_tensor.buffer());

        // Update reader kernel semaphore addresses
        auto& reader_runtime_args_by_core =
            tt::tt_metal::GetRuntimeArgs(program, shared_variables.send_reader_kernel_id);
        for (const auto& core : shared_variables.cores) {
            auto& args = reader_runtime_args_by_core[core.x][core.y];
            args[0] = shared_variables.semaphores[0].address();
            args[1] = shared_variables.semaphores[1].address();
            args[2] = shared_variables.semaphores[2].address();
        }

        // Update writer kernel args based on device role
        // LEAF: sends to intermediate_tensor[0], semaphore[0]
        // ROOT3: sends to intermediate_tensor[1], semaphore[1]
        // ROOT2: sends to intermediate_tensor[2], semaphore[2]
        uint32_t tensor_idx = shared_variables.is_mesh_leaf_device    ? 0
                              : shared_variables.is_mesh_root3_device ? 1
                              : shared_variables.is_mesh_root2_device ? 2
                                                                      : 0;

        // Update writer kernel args based on device role
        auto& worker_writer_args_by_core =
            tt::tt_metal::GetRuntimeArgs(program, shared_variables.send_worker_writer_kernel_id);

        if (!shared_variables.is_mesh_root1_device) {
            // Non-ROOT1: update destination addresses for fabric sends
            uint32_t dst_l1_addr = intermediate_tensors[tensor_idx].buffer()->address();
            uint32_t dst_sem_addr = shared_variables.semaphores[tensor_idx].address();

            for (const auto& core : shared_variables.cores) {
                auto& args = worker_writer_args_by_core[core.x][core.y];
                args[4] = dst_l1_addr;
                args[5] = dst_sem_addr;
            }
        } else {
            // ROOT1: update output base address for gather
            uint32_t output_base_addr = output_tensor.buffer()->address();

            for (const auto& core : shared_variables.cores) {
                auto& args = worker_writer_args_by_core[core.x][core.y];
                args[6] = output_base_addr;
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl
