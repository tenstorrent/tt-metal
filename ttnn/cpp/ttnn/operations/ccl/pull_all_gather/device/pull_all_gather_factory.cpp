// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pull_all_gather_factory.hpp"

#include <bit>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl {

namespace {

// Concatenate arg vectors in declaration order.
template <typename... Vs>
std::vector<uint32_t> concat(Vs&&... vs) {
    std::vector<uint32_t> out;
    (out.insert(out.end(), std::begin(vs), std::end(vs)), ...);
    return out;
}

}  // namespace

PullAllGatherFactory::cached_mesh_workload_t PullAllGatherFactory::create_mesh_workload(
    const PullAllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const PullAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(*operation_attributes.sub_core_grid);
    }
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Peers must not multicast into an output buffer this device has not
    // allocated yet, so all devices reach the kernel before any data moves.
    const bool l1_small = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0;
    auto barrier_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, available_cores, 0, l1_small ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, output_tensor, barrier_sem);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

PullAllGatherFactory::cached_program_t PullAllGatherFactory::create_at(
    const PullAllGatherParams& args,
    const ttnn::MeshCoordinate& sender_device_coord,
    const PullAllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem) {
    namespace m2 = tt::tt_metal::experimental;

    const auto& input_tensor = tensor_args.input_tensor;
    auto* mesh_device = input_tensor.device();

    // ---- Sizes, all derived from the tensors and the fabric ----
    const auto& input_spec = input_tensor.tensor_spec();
    const auto& tile = input_spec.tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t tile_bytes = input_spec.compute_page_size_bytes();

    const auto& shape = input_spec.logical_shape();
    const uint32_t rows = shape[shape.rank() - 2];  // M, the gather dim
    const uint32_t cols = shape[shape.rank() - 1];  // N
    TT_FATAL(rows % tile_h == 0 && cols % tile_w == 0, "Shape must be tile aligned");

    const uint32_t tile_rows = rows / tile_h;  // this device's block
    const uint32_t tile_cols = cols / tile_w;
    const uint32_t tiles_per_device = tile_rows * tile_cols;

    // Both height sharded, so each shard is one contiguous byte range. The two
    // shard specs are independent: only the output's has to divide the block
    // (validate() checked it), and the input's last shard per block may be
    // ragged.
    const auto& in_shard_shape = input_spec.memory_config().nd_shard_spec()->shard_shape;
    const auto& out_shard_shape = args.output_mem_config.nd_shard_spec()->shard_shape;
    const uint32_t in_shard_tiles = (in_shard_shape[-2] / tile_h) * tile_cols;
    const uint32_t out_shard_tiles = (out_shard_shape[-2] / tile_h) * tile_cols;
    const uint32_t in_shard_bytes = in_shard_tiles * tile_bytes;
    const uint32_t out_shard_bytes = out_shard_tiles * tile_bytes;
    const uint32_t block_bytes = tiles_per_device * tile_bytes;

    // Entry = the largest chunk the rule can produce. Nothing is tile
    // quantised: a chunk is a byte range inside one shard on each side.
    const uint32_t bytes_per_dma_txn =
        std::min<size_t>(args.max_payload_bytes, std::min(in_shard_bytes, out_shard_bytes));

    // The same walk both kernels run, to get the chunk count.
    auto txn_bytes_at = [&](uint32_t cursor) {
        const uint32_t in_end = std::min((cursor / in_shard_bytes + 1) * in_shard_bytes, block_bytes);
        const uint32_t out_left = out_shard_bytes - (cursor % out_shard_bytes);
        return std::min(bytes_per_dma_txn, std::min(in_end - cursor, out_left));
    };
    uint32_t txns_per_device = 0;
    for (uint32_t cursor = 0; cursor < block_bytes; cursor += txn_bytes_at(cursor)) {
        ++txns_per_device;
    }

    // ---- ProgramSpec ----
    const m2::DFBSpecName kPayloadDfb{"payload"};
    const m2::KernelSpecName kProducer{"pull_all_gather_producer"};
    const m2::KernelSpecName kSender{"pull_all_gather_sender"};
    const m2::TensorParamName kInputTensor{"input_tensor"};
    const m2::TensorParamName kOutputTensor{"output_tensor"};
    const m2::ScratchpadSpecName kFabricRequests{"fabric_requests"};
    constexpr m2::NodeCoord kWorkerNode{0, 0};

    // How many request sets: one per packet state the sender keeps live -- the
    // payload multicast and the completion atomic. Two NocSendTypes, two sticky
    // headers; set-state for one would clobber the other. Independent of
    // topology.
    constexpr uint32_t kNumRequestSets = 2;
    // Slots within a set: one per route, since a route needs its own packet
    // header. 2 for 1D, 4 for 2D, so 4 and 8 slots in total. Same expression
    // the kernel's FabricPullRequestSet uses, so host and kernel cannot
    // disagree on the count.
    constexpr uint32_t kMaxRoutes = tt::tt_fabric::fabric_max_routes<topology>;
    // One route block per slot, always kMaxRoutes of them so the vararg count
    // is fixed; blocks past num_routes are zero and never read. A forwarding
    // route is h[0..3] + port + dst_dev_id + dst_mesh_id; a peer route is the
    // mask alone.
    constexpr uint32_t kRouteWords = tt::tt_fabric::is_forwarding_topology(topology) ? 7 : 1;
    // sizeof(FabricPullRequestSet<PACKET_HEADER_TYPE, kMaxRoutes>), which the
    // host cannot write directly: PACKET_HEADER_TYPE is a kernel-side define,
    // so only its size is available here. The helper lives beside the struct in
    // fabric_edm_types.hpp, so the layout is not spelled out twice.
    const uint32_t request_set_bytes = tt::tt_fabric::fabric_pull_request_set_bytes(
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()), kMaxRoutes);
    // device_idx, num_peers | route block | sem addr, x, y. A forwarding
    // topology's block is num_routes plus the fixed kMaxRoutes route words; a
    // peer topology's is the mask alone, which is already the whole route.
    constexpr uint32_t kRouteArgWords =
        tt::tt_fabric::is_forwarding_topology(topology) ? 1 + kMaxRoutes * kRouteWords : 1;
    constexpr uint32_t kSenderRuntimeArgs = 2 + kRouteArgWords + 3;

    m2::KernelSpec producer{
        .unique_id = kProducer,
        .source = "ttnn/cpp/ttnn/operations/ccl/pull_all_gather/device/kernels/producer.cpp",
        .num_threads = args.num_producers,
        // Implicit sync stays on here: the producer wants the TRID path.
    };
    producer.dfb_bindings = {m2::ProducerOf(kPayloadDfb, "payload")};
    producer.tensor_bindings = {{.tensor_parameter_name = kInputTensor, .accessor_name = "input_tensor"}};
    // No per-producer count: each thread walks the whole chunk sequence and
    // acts on the entries it owns, so the stride is all it needs.
    producer.compile_time_args = {
        {"txns_per_device", txns_per_device},
        {"num_producers", args.num_producers},
        {"bytes_per_dma_txn", bytes_per_dma_txn},
        {"in_shard_bytes", in_shard_bytes},
        {"in_shard_tiles", in_shard_tiles},
        {"out_shard_bytes", out_shard_bytes},
        {"block_bytes", block_bytes},
    };

    m2::KernelSpec sender{
        .unique_id = kSender,
        .source = "ttnn/cpp/ttnn/operations/ccl/pull_all_gather/device/kernels/sender.cpp",
        .num_threads = 1,
        .hw_config = m2::DataMovementGen2Config{.disable_dfb_implicit_sync_for = {kPayloadDfb}},
    };
    sender.dfb_bindings = {m2::ConsumerOf(kPayloadDfb, "payload")};
    // Request slots only. The transaction-counter bank is reserved L1, so it
    // has no scratchpad.
    sender.scratchpad_bindings = {{.scratchpad_spec_name = kFabricRequests, .accessor_name = "fabric_requests"}};
    sender.tensor_bindings = {{.tensor_parameter_name = kOutputTensor, .accessor_name = "output_tensor"}};
    // Hop counts are per-device, so they are runtime args; only shape constants
    // are compile time.
    sender.compile_time_args = {
        {"txns_per_device", txns_per_device},
        {"bytes_per_dma_txn", bytes_per_dma_txn},
        {"in_shard_bytes", in_shard_bytes},
        {"out_shard_bytes", out_shard_bytes},
        {"out_shard_tiles", out_shard_tiles},
        {"block_bytes", block_bytes},
        {"tiles_per_device", tiles_per_device},
    };
    sender.advanced_options.num_runtime_varargs = kSenderRuntimeArgs;

    const m2::ProgramSpec program_spec{
        .name = "pull_all_gather",
        .kernels = {producer, sender},
        .dataflow_buffers = {{
            .unique_id = kPayloadDfb,
            .entry_size = bytes_per_dma_txn,  // a DMA transfer, not a tensor page
            .num_entries = args.dfb_depth,
            .data_format_metadata = input_spec.data_format(),
        }},
        .scratchpads = {{
            .unique_id = kFabricRequests,
            .size_per_node = kNumRequestSets * request_set_bytes,
        }},
        .tensor_parameters =
            {
                {.unique_id = kInputTensor, .spec = input_spec},
                {.unique_id = kOutputTensor, .spec = output_tensor.tensor_spec()},
            },
        .work_units = {{
            .name = "pull_all_gather_worker",
            .kernels = {kProducer, kSender},
            .target_nodes = kWorkerNode,
        }},
    };

    auto program = m2::MakeProgramFromSpec(*mesh_device, program_spec);

    // ---- Per-device runtime args ----
    const uint32_t device_idx =
        ::ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, sender_device_coord, args.cluster_axis);

    std::vector<uint32_t> route_args;
    uint32_t num_routes = 0;

    if constexpr (!tt::tt_fabric::is_forwarding_topology(topology)) {
        // Every peer, named by fabric node id -- the mask means the same set of
        // devices whichever device sends it, which is why it names nodes rather
        // than the DE's queue indices. Our own bit stays clear: chip multicast
        // excludes the source, and our replica is include_self.
        const auto my_node = mesh_device->get_fabric_node_id(sender_device_coord);
        uint32_t peer_mask = 0;
        for (const auto& coord : ttnn::MeshCoordinateRange(mesh_device->shape())) {
            const auto node = mesh_device->get_fabric_node_id(coord);
            if (node == my_node) {
                continue;
            }
            TT_FATAL(
                static_cast<uint32_t>(node.chip_id) < 32,
                "Peer mask holds 32 nodes; node {} does not fit",
                node.chip_id);
            peer_mask |= (1u << static_cast<uint32_t>(node.chip_id));
        }
        TT_FATAL(
            static_cast<uint32_t>(std::popcount(peer_mask)) == args.peer_count(),
            "Mask must name every peer exactly once");

        route_args = {peer_mask};
        num_routes = 1;
    } else {
        // Hop counts per direction, plus the physical E/W/N/S slot each forwards
        // through.
        uint32_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
        std::optional<MeshCoordinate> e_coord, w_coord, n_coord, s_coord;

        for (uint32_t axis = 0; axis < 2; ++axis) {
            if (args.axis_num_devices[axis] <= 1) {
                continue;  // inactive axis
            }
            const auto axis_topology = args.axis_topology[axis];

            auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
                args.axis_num_devices[axis], sender_device_coord[axis], axis_topology, /*static_alternate=*/false);
            auto fwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
                input_tensor, sender_device_coord, 1, axis_topology, axis);
            auto bwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
                input_tensor, sender_device_coord, -1, axis_topology, axis);

            // axis 1 -> (E = fwd, W = bwd); axis 0 -> (S = fwd, N = bwd)
            if (axis == 1) {
                e_hops = fwd_hops;
                w_hops = bwd_hops;
                e_coord = fwd_coord;
                w_coord = bwd_coord;
            } else {
                s_hops = fwd_hops;
                n_hops = bwd_hops;
                s_coord = fwd_coord;
                n_coord = bwd_coord;
            }
        }
        // The two lines cover this device's own row; each rect covers `spine` rows
        // beyond it, every one of them (e + w + 1) wide. On 1D both spines are zero
        // and this reduces to e + w.
        TT_FATAL(
            (e_hops + w_hops) + (n_hops + s_hops) * (e_hops + w_hops + 1) == args.peer_count(),
            "Routes must cover every peer exactly once");

        // Physical direction each neighbour sits in; depends on mesh position.
        const auto sender_node = mesh_device->get_fabric_node_id(sender_device_coord);
        auto physical_slot = [&](const std::optional<MeshCoordinate>& neighbor) -> uint32_t {
            if (!neighbor.has_value()) {
                return 0;
            }
            const auto dir =
                tt::tt_fabric::get_eth_forwarding_direction(sender_node, mesh_device->get_fabric_node_id(*neighbor));
            TT_FATAL(
                dir.has_value() && static_cast<uint32_t>(*dir) < tt::tt_fabric::eth_chan_directions::Z,
                "Expected a cardinal E/W/N/S forwarding direction");
            return static_cast<uint32_t>(*dir);
        };
        // Not the route's final destination -- there is none for a multicast.
        // dst_dev_id / dst_mesh_id become packet_header->dst_start_node_id, the
        // anchor the E/W/N/S hop counts extend from, so it is the chip the packet
        // enters on: this route's first hop.
        auto dst_ids = [&](const std::optional<MeshCoordinate>& neighbor) -> std::pair<uint32_t, uint32_t> {
            if (!neighbor.has_value()) {
                return {0, 0};
            }
            const auto node = mesh_device->get_fabric_node_id(*neighbor);
            return {static_cast<uint32_t>(node.chip_id), static_cast<uint32_t>(*node.mesh_id)};
        };

        const uint32_t e_dir = physical_slot(e_coord), w_dir = physical_slot(w_coord);
        const uint32_t n_dir = physical_slot(n_coord), s_dir = physical_slot(s_coord);

        // Up to four routes: the E and W lines along this row, and the N and S
        // rects, each fanning E/W within its own spine. A zero hop count
        // contributes no route, so 1D fills two and a line endpoint fills one.
        auto add_route =
            [&](uint32_t spine_hops, uint32_t spine_dir, bool fan_out, const std::optional<MeshCoordinate>& first_hop) {
                if (spine_hops == 0) {
                    return;
                }
                uint32_t h[4] = {};
                h[spine_dir] = spine_hops;
                if (fan_out) {
                    if (e_hops > 0) {
                        h[e_dir] = e_hops;
                    }
                    if (w_hops > 0) {
                        h[w_dir] = w_hops;
                    }
                }
                const auto [dst_dev, dst_mesh] = dst_ids(first_hop);
                route_args.insert(route_args.end(), {h[0], h[1], h[2], h[3], spine_dir, dst_dev, dst_mesh});
                ++num_routes;
            };
        add_route(e_hops, e_dir, /*fan_out=*/false, e_coord);
        add_route(w_hops, w_dir, /*fan_out=*/false, w_coord);
        add_route(s_hops, s_dir, /*fan_out=*/true, s_coord);
        add_route(n_hops, n_dir, /*fan_out=*/true, n_coord);

        TT_FATAL(num_routes > 0 && num_routes <= kMaxRoutes, "Need 1..{} routes, derived {}", kMaxRoutes, num_routes);
        route_args.insert(route_args.begin(), num_routes);
        route_args.resize(1 + kMaxRoutes * kRouteWords, 0);  // fixed vararg count
    }

    const auto barrier_node = mesh_device->worker_core_from_logical_core(kWorkerNode);

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {{
        .kernel = kSender,
        .advanced_options =
            {.runtime_varargs =
                 {{kWorkerNode,
                   concat(
                       std::vector<uint32_t>{device_idx, args.peer_count()},
                       route_args,
                       std::vector<uint32_t>{
                           static_cast<uint32_t>(barrier_sem.address()), barrier_node.x, barrier_node.y})}}},
    }};
    run_args.tensor_args = {
        {kInputTensor, std::cref(input_tensor)},
        {kOutputTensor, std::cref(output_tensor)},
    };
    m2::SetProgramRunArgs(program, run_args);

    return {std::move(program), shared_variables_t{barrier_sem, device_idx, std::move(route_args)}};
}

// On a cache hit the tensors may have moved. Production patches addresses into
// cached arg vectors by index; Metal 2.0 has no index to patch, since
// SetProgramRunArgs is what binds them -- so the override re-runs it.
void PullAllGatherFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const PullAllGatherParams& args,
    const PullAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    namespace m2 = tt::tt_metal::experimental;

    auto* mesh_device = tensor_args.input_tensor.device();
    const auto barrier_node = mesh_device->worker_core_from_logical_core(m2::NodeCoord{0, 0});

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        m2::ProgramRunArgs run_args;
        run_args.kernel_run_args = {{
            .kernel = m2::KernelSpecName{"pull_all_gather_sender"},
            .advanced_options =
                {.runtime_varargs =
                     {{m2::NodeCoord{0, 0},
                       concat(
                           std::vector<uint32_t>{shared_vars.device_idx, args.peer_count()},
                           shared_vars.route_args,
                           std::vector<uint32_t>{
                               static_cast<uint32_t>(shared_vars.barrier_sem.address()),
                               barrier_node.x,
                               barrier_node.y})}}},
        }};
        run_args.tensor_args = {
            {m2::TensorParamName{"input_tensor"}, std::cref(tensor_args.input_tensor)},
            {m2::TensorParamName{"output_tensor"}, std::cref(output_tensor)},
        };
        m2::SetProgramRunArgs(program, run_args);
    }
}

}  // namespace ttnn::operations::ccl
