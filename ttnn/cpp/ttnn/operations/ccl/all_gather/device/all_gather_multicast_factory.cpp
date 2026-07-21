// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_multicast_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <set>

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

namespace {

constexpr uint32_t explicit_path_word_count = 5;

struct PackedFabricPath {
    uint32_t length = 0;
    uint32_t escape_hop = 0;
    std::array<uint32_t, explicit_path_word_count> words{};
};

tt::tt_fabric::eth_chan_directions opposite_direction(tt::tt_fabric::eth_chan_directions direction) {
    using tt::tt_fabric::eth_chan_directions;
    switch (direction) {
        case eth_chan_directions::EAST: return eth_chan_directions::WEST;
        case eth_chan_directions::WEST: return eth_chan_directions::EAST;
        case eth_chan_directions::NORTH: return eth_chan_directions::SOUTH;
        case eth_chan_directions::SOUTH: return eth_chan_directions::NORTH;
        default: TT_THROW("A same-mesh multicast path cannot use the Z direction");
    }
}

uint32_t bank_owned_rows_per_run(const AllGatherReceiverPolicy& policy, uint32_t max_rows, uint32_t pages_per_bank) {
    if (policy.bank_owned_run_policy == BankOwnedRunPolicy::MaxTail) {
        return max_rows;
    }
    for (uint32_t rows = std::min(max_rows, pages_per_bank); rows > 0; --rows) {
        if (pages_per_bank % rows == 0) {
            return rows;
        }
    }
    return 1;
}

uint32_t receiver_l1_drain_risc_count(const AllGatherReceiverPolicy& policy, const Tensor& input_tensor) {
    if (policy.drain_risc_count == 0) {
        // Dual-RISC drain raises the isolated FP8 receiver from 31.5 to 37.1 GB/s,
        // while BF16 is already sender/Fabric limited and gains less than the 3%
        // acceptance threshold. Explicit 1/2 values remain test-only A/B controls.
        return input_tensor.dtype() == ttnn::DataType::FP8_E4M3 ? 2 : 1;
    }
    return policy.drain_risc_count;
}

uint32_t receiver_l1_batch_rows(const AllGatherReceiverPolicy& policy, uint32_t max_rows) {
    if (policy.batch_rows == 0) {
        return max_rows;
    }
    TT_FATAL(
        policy.batch_rows <= max_rows,
        "receiver batch of {} rows exceeds packet limit of {} rows",
        policy.batch_rows,
        max_rows);
    return policy.batch_rows;
}

struct ReceiverL1Plan {
    bool eligible = false;
    std::string_view rejection_reason = "uninitialized receiver plan";
    uint32_t active_axis = 0;
    uint32_t num_links = 0;
    uint32_t required_worker_cores = 0;
    uint32_t available_worker_cores = 0;
    uint32_t pages_per_packet = 0;
    uint32_t slot_count = 0;
    uint64_t staging_bytes = 0;
    uint32_t staging_base = 0;
    uint64_t staging_end = 0;
    uint32_t control_semaphore_count = 0;
    uint64_t control_semaphore_bytes_per_core = 0;
};

bool uses_explicit_ring_path(const AllGatherParams& attrs) {
    const bool one_active_axis = (attrs.axis_num_devices[0] > 1) != (attrs.axis_num_devices[1] > 1);
    const uint32_t active_axis = attrs.axis_num_devices[0] > 1 ? 0 : 1;
    return tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig()) && one_active_axis &&
           attrs.axis_topology[active_axis] == tt::tt_fabric::Topology::Ring;
}

// Build the bounded host-side resource and mapping proof used by automatic
// dispatch. Unsupported mappings return a concrete rejection reason and keep
// the existing direct-scatter implementation. In particular, auto mode must
// never select the receiver and then fail because its extra mirrored cores or
// staging slots do not fit.
ReceiverL1Plan make_receiver_l1_plan(
    const AllGatherParams& attrs, const AllGatherInputs& tensor_args, const Tensor& output_tensor) {
    ReceiverL1Plan plan;
    const auto reject = [&plan](std::string_view reason) {
        plan.rejection_reason = reason;
        return plan;
    };
    const auto& input = tensor_args.input_tensor;
    const auto shape = input.padded_shape();
    const auto output_shape = output_tensor.padded_shape();
    int32_t dim = attrs.dim;
    if (dim < 0) {
        dim += shape.rank();
    }
    if (!tensor_args.persistent_output_tensor.has_value()) {
        return reject("receiver requires a persistent output tensor");
    }
    if (input.layout() != ttnn::ROW_MAJOR_LAYOUT || output_tensor.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        return reject("receiver requires row-major input and output tensors");
    }
    if (shape.rank() != 4 || output_shape.rank() != 4) {
        return reject("receiver currently requires rank-4 input and output tensors");
    }
    if (dim != 2 || shape[1] != 1) {
        return reject("receiver currently requires a height gather with shape dimension 1 equal to one");
    }
    const bool one_active_axis = (attrs.axis_num_devices[0] > 1) != (attrs.axis_num_devices[1] > 1);
    if (!one_active_axis) {
        return reject("receiver currently requires exactly one active mesh axis");
    }
    plan.active_axis = attrs.axis_num_devices[0] > 1 ? 0 : 1;
    const auto active_topology = attrs.axis_topology[plan.active_axis];
    if (active_topology != tt::tt_fabric::Topology::Linear && active_topology != tt::tt_fabric::Topology::Ring) {
        return reject("receiver currently requires linear or single-axis ring topology");
    }
    if (attrs.num_devices > 8) {
        return reject("receiver currently supports at most eight devices on the active axis");
    }
    const bool supported_batch_selection =
        shape[0] == 1 ||
        (attrs.batch_slice_idx.has_value() && attrs.batch_slice_idx.value() < shape[0] && output_shape[0] == 1);
    if (!supported_batch_selection) {
        return reject("multi-batch receiver input requires one valid batch_slice_idx and a single-batch output");
    }
    const bool supported_gather_extent =
        !attrs.valid_gather_extent.has_value() ||
        (attrs.valid_gather_extent.value() > 0 && attrs.valid_gather_extent.value() <= shape[2]);
    if (!supported_gather_extent) {
        return reject("receiver valid_gather_extent must select a non-empty leading height prefix");
    }
    if (input.buffer()->buffer_type() != tt::tt_metal::BufferType::DRAM ||
        output_tensor.buffer()->buffer_type() != tt::tt_metal::BufferType::DRAM) {
        return reject("receiver currently requires DRAM input and output buffers");
    }
    // The sender reader already uses TensorAccessor for every source page, so its addressing is
    // valid for interleaved and ND-sharded DRAM inputs alike. Receiver staging changes only the
    // remote destination: the persistent output must remain interleaved so every receiver can drain
    // matched full pages to the same replicated layout. The page-geometry check below rejects width
    // or block sharding that does not preserve one complete row per source page.
    if (output_tensor.buffer()->buffer_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return reject("receiver currently requires an interleaved output buffer");
    }
    const uint32_t input_page_size = input.buffer()->aligned_page_size();
    if (input_page_size == 0 || input_page_size != output_tensor.buffer()->aligned_page_size() ||
        input.buffer()->page_size() != output_tensor.buffer()->page_size()) {
        return reject("receiver requires matched non-empty input and output page geometry");
    }
    if (input_page_size > attrs.packet_size) {
        return reject("one receiver input page exceeds the Fabric payload limit");
    }

    const uint32_t links0 = attrs.axis_num_links[0];
    const uint32_t links1 = attrs.axis_num_links[1];
    plan.num_links = std::min(links0 > 0 ? links0 : links1, links1 > 0 ? links1 : links0);
    if (plan.num_links == 0) {
        return reject("receiver requires at least one active Fabric link");
    }
    if (attrs.receiver_policy.interleaved_bank_receivers) {
        if (!attrs.receiver_policy.bank_owned_links) {
            return reject("interleaved bank receivers require bank-owned links");
        }
        if (plan.num_links != 2) {
            return reject("interleaved bank receivers require two active links");
        }
        if (input.device()->num_dram_channels() != 8) {
            return reject("interleaved bank receivers require eight DRAM banks");
        }
    }
    const uint32_t receiver_cores_per_link = attrs.receiver_policy.interleaved_bank_receivers ? 4 : 1;
    plan.required_worker_cores = (1 + receiver_cores_per_link) * plan.num_links;
    auto subdevice_id = attrs.subdevice_id.value_or(input.device()->get_sub_device_ids().at(0));
    auto available_cores = input.device()->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (attrs.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(attrs.sub_core_grid.value());
    }
    plan.available_worker_cores = available_cores.num_cores();
    if (plan.available_worker_cores < plan.required_worker_cores) {
        return reject("receiver sub-core grid does not contain the required sender and receiver workers per link");
    }

    // Receiver payloads live in ordinary L1, but all of its control state must
    // live in L1-small.  Prove the complete mirrored semaphore allocation here,
    // before dispatch, rather than silently fragmenting or overlapping ordinary
    // L1.  Global semaphores consume one allocator-aligned slot per participating
    // core even though their logical value is one uint32_t.
    const auto& allocator = input.device()->allocator();
    const uint64_t l1_small_bank_size = allocator->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    if (l1_small_bank_size == 0) {
        return reject("receiver requires an L1-small region for control semaphores");
    }
    const uint32_t semaphore_alignment = allocator->get_alignment(tt::tt_metal::BufferType::L1_SMALL);
    if (semaphore_alignment == 0) {
        return reject("receiver L1-small semaphore alignment is zero");
    }
    plan.control_semaphore_count =
        1 + attrs.num_devices + (attrs.receiver_policy.bank_owned_links ? attrs.num_devices : 0) +
        2 * receiver_cores_per_link +
        (receiver_l1_drain_risc_count(attrs.receiver_policy, input) == 2 ? 1 : 0);  // barrier + receiver controls
    plan.control_semaphore_bytes_per_core = static_cast<uint64_t>(plan.control_semaphore_count) * semaphore_alignment;
    if (plan.control_semaphore_bytes_per_core > l1_small_bank_size) {
        return reject("receiver control semaphores exceed the configured L1-small bank");
    }

    const uint32_t max_pages_per_packet = std::max(1u, static_cast<uint32_t>(attrs.packet_size) / input_page_size);
    plan.pages_per_packet = receiver_l1_batch_rows(attrs.receiver_policy, max_pages_per_packet);
    const uint64_t slot_stride = static_cast<uint64_t>(input_page_size) * plan.pages_per_packet;
    plan.staging_base = input.device()->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    if (plan.staging_base >= input.device()->l1_size_per_core()) {
        return reject("receiver ordinary-L1 allocator base is outside the worker L1 region");
    }
    const uint64_t available_staging_bytes = input.device()->l1_size_per_core() - plan.staging_base;
    plan.slot_count = attrs.receiver_policy.slot_count;
    if (plan.slot_count == 0) {
        plan.slot_count = available_staging_bytes / (static_cast<uint64_t>(attrs.num_devices) * slot_stride);
    }
    if (plan.slot_count == 0) {
        return reject("receiver ordinary-L1 region cannot hold one complete source-slot set");
    }
    plan.staging_bytes = static_cast<uint64_t>(attrs.num_devices) * plan.slot_count * slot_stride;
    plan.staging_end = static_cast<uint64_t>(plan.staging_base) + plan.staging_bytes;
    if (plan.staging_end > input.device()->l1_size_per_core()) {
        return reject("requested receiver slot window exceeds the ordinary-L1 region");
    }

    plan.eligible = true;
    plan.rejection_reason = {};
    return plan;
}

bool auto_receiver_l1_path_is_preferred(const AllGatherParams& attrs, const Tensor& /*input*/) {
    const bool active_axis_is_ring =
        (attrs.axis_num_devices[0] > 1 && attrs.axis_topology[0] == tt::tt_fabric::Topology::Ring) ||
        (attrs.axis_num_devices[1] > 1 && attrs.axis_topology[1] == tt::tt_fabric::Topology::Ring);
    if (!active_axis_is_ring) {
        return true;
    }

    return attrs.receiver_policy.bank_owned_links;
}

bool use_receiver_l1_path(
    const AllGatherParams& attrs, const AllGatherInputs& tensor_args, const Tensor& output_tensor) {
    const auto plan = make_receiver_l1_plan(attrs, tensor_args, output_tensor);

    switch (attrs.receiver_policy.test_mode) {
        case ReceiverL1TestMode::Auto:
            if (!plan.eligible) {
                log_debug(tt::LogOp, "Receiver-L1 all-gather fallback: {}", plan.rejection_reason);
                return false;
            }
            if (!auto_receiver_l1_path_is_preferred(attrs, tensor_args.input_tensor)) {
                log_debug(tt::LogOp, "Receiver-L1 all-gather fallback: direct path is preferred for this ring case");
                return false;
            }
            return true;
        case ReceiverL1TestMode::ForceDirect: return false;
        case ReceiverL1TestMode::ForceReceiver:
            TT_FATAL(
                plan.eligible,
                "Forced receiver-L1 all-gather is not eligible for this tensor and operation: {}",
                plan.rejection_reason);
            return true;
    }
    return false;
}

}  // namespace

bool should_auto_select_receiver_l1_path(
    const AllGatherParams& operation_attributes, const AllGatherInputs& tensor_args, const Tensor& output_tensor) {
    return make_receiver_l1_plan(operation_attributes, tensor_args, output_tensor).eligible &&
           auto_receiver_l1_path_is_preferred(operation_attributes, tensor_args.input_tensor);
}

AllGatherMulticastFactory::cached_mesh_workload_t AllGatherMulticastFactory::create_mesh_workload(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(operation_attributes.sub_core_grid.value());
    }
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Resolve the complete configured-capacity proof before allocating even
    // the common barrier.  This keeps force_receiver failures host-side and
    // deterministic when the L1-small region itself cannot hold the full
    // protocol. Runtime allocator pressure remains enforced by the allocator,
    // as it is for the existing direct path and cached global semaphores.
    const bool receiver_l1_mode = use_receiver_l1_path(operation_attributes, tensor_args, output_tensor);

    // Kernel needs to wait to receive all remote data before exiting, and in some cases needs to wait
    // for all remote devices to be ready before beginning operation.
    // Since Fabric doesn't provide such capability within kernels, we need to manually sync using global semaphores.
    // Allocate the semaphore in L1_SMALL to avoid fragmenting the larger L1 memory pool.
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and reduce headroom for subsequent op "
            "allocations. Configure an L1_SMALL region to mitigate this.");
    }
    auto barrier_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    std::vector<tt::tt_metal::GlobalSemaphore> receiver_control_sems;
    if (receiver_l1_mode) {
        // Remote-consumption credit, local receiver-consumed sequence,
        // forward/reader payload-produced sequences, and backward/writer
        // payload-produced sequences,
        // present only for the bank-owned path. Generic routing uses the forward
        // sequence for both workers and does not pay this L1-small cost.
        // The final entry is present only for dual-RISC receiver drain.
        const uint32_t receiver_cores_per_link =
            operation_attributes.receiver_policy.interleaved_bank_receivers ? 4 : 1;
        const uint32_t receiver_control_sem_count =
            operation_attributes.num_devices +
            (operation_attributes.receiver_policy.bank_owned_links ? operation_attributes.num_devices : 0) +
            2 * receiver_cores_per_link +
            (receiver_l1_drain_risc_count(operation_attributes.receiver_policy, tensor_args.input_tensor) == 2 ? 1 : 0);
        receiver_control_sems.reserve(receiver_control_sem_count);
        for (uint32_t i = 0; i < receiver_control_sem_count; ++i) {
            receiver_control_sems.push_back(
                ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
        }
    }
    log_debug(tt::LogOp, "Semaphore allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, output_tensor, barrier_sem, receiver_control_sems);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherMulticastFactory::cached_program_t AllGatherMulticastFactory::create_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem,
    const std::vector<tt::tt_metal::GlobalSemaphore>& receiver_control_sems) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto* mesh_device = input_tensor.device();
    tt::tt_metal::Program program{};

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    ////////////////////////////////////////////////////////////////

    const uint32_t num_devices = operation_attributes.num_devices;
    uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    // Compute hops + neighbors for each mesh axis.
    // Each axis ∈ {0, 1} contributes a forward/backward pair: axis 1 -> (E=fwd, W=bwd),
    // axis 0 -> (S=fwd, N=bwd). In 1D only one axis is active; in 2D both can be.
    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());

    std::optional<MeshCoordinate> e_coord, w_coord, n_coord, s_coord;
    uint32_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
    bool ew_load_balance = false;
    bool ns_load_balance = false;

    for (uint32_t axis = 0; axis < 2; ++axis) {
        const uint32_t axis_size = operation_attributes.axis_num_devices[axis];
        const bool is_axis_active = axis_size > 1;
        if (!is_axis_active) {
            continue;
        }

        const auto axis_topology = operation_attributes.axis_topology[axis];
        const uint32_t axis_index = sender_device_coord[axis];
        auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
            axis_size, axis_index, axis_topology, /*static_alternate=*/false);
        auto fwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, 1, axis_topology, axis);
        auto bwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, -1, axis_topology, axis);

        // A load-balancing technique (alternating between two imbalanced routes) is used
        // in even-sized rings.
        const bool axis_load_balance = tt::tt_fabric::is_ring_or_torus(axis_topology) && (axis_size % 2 == 0);
        if (axis == 1) {
            e_hops = fwd_hops;
            w_hops = bwd_hops;
            e_coord = fwd_coord;
            w_coord = bwd_coord;
            ew_load_balance = axis_load_balance;
        } else {
            s_hops = fwd_hops;
            n_hops = bwd_hops;
            s_coord = fwd_coord;
            n_coord = bwd_coord;
            ns_load_balance = axis_load_balance;
        }
    }
    const uint32_t active_axis = operation_attributes.axis_num_devices[0] > 1 ? 0 : 1;
    const bool use_explicit_ring_path = uses_explicit_ring_path(operation_attributes);
    auto build_explicit_path = [&](int step, uint32_t target_count) {
        PackedFabricPath path;
        path.length = target_count;
        TT_FATAL(
            target_count <= explicit_path_word_count * 8,
            "Explicit Fabric ring path of {} hops exceeds the {}-hop host encoding",
            target_count,
            explicit_path_word_count * 8);
        if (target_count == 0) {
            return path;
        }

        std::vector<tt::tt_fabric::eth_chan_directions> movements;
        movements.reserve(target_count);
        auto previous_coord = sender_device_coord;
        uint32_t logical_index = device_idx;
        for (uint32_t hop = 0; hop < target_count; ++hop) {
            auto next_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
                input_tensor, previous_coord, step, tt::tt_fabric::Topology::Ring, active_axis);
            TT_FATAL(next_coord.has_value(), "Explicit Fabric ring path is missing logical neighbor {}", hop + 1);
            const auto previous_node = mesh_device->get_fabric_node_id(previous_coord);
            const auto next_node = mesh_device->get_fabric_node_id(*next_coord);
            const auto direction = tt::tt_fabric::get_eth_forwarding_direction(previous_node, next_node);
            TT_FATAL(direction.has_value(), "No Fabric route from {} to ring neighbor {}", previous_node, next_node);
            movements.push_back(*direction);
            const uint32_t logical_step = step > 0 ? 1 : num_devices - 1;
            const uint32_t next_logical_index = (logical_index + logical_step) % num_devices;
            const bool crosses_dateline = (logical_index == num_devices - 1 && next_logical_index == 0) ||
                                          (logical_index == 0 && next_logical_index == num_devices - 1);
            if (crosses_dateline && hop + 1 < target_count) {
                // Store hop+1 so zero continues to mean that this path never
                // needs the Fabric escape VC. The receiving router switches
                // the packet after it traverses the dateline link.
                path.escape_hop = hop + 1;
            }
            logical_index = next_logical_index;
            previous_coord = *next_coord;
        }

        for (uint32_t hop = 0; hop < target_count; ++hop) {
            uint8_t command = 1u << static_cast<uint8_t>(opposite_direction(movements[hop]));
            if (hop + 1 < target_count) {
                command |= 1u << static_cast<uint8_t>(movements[hop + 1]);
            }
            path.words[hop / 8] |= static_cast<uint32_t>(command) << ((hop % 8) * 4);
        }
        return path;
    };

    PackedFabricPath reader_path;
    PackedFabricPath reader_path_alt;
    PackedFabricPath writer_path;
    PackedFabricPath writer_path_alt;
    if (use_explicit_ring_path) {
        const uint32_t forward_hops = active_axis == 0 ? s_hops : e_hops;
        const uint32_t backward_hops = active_axis == 0 ? n_hops : w_hops;
        reader_path = build_explicit_path(1, forward_hops);
        reader_path_alt = build_explicit_path(1, backward_hops);
        writer_path = build_explicit_path(-1, backward_hops);
        writer_path_alt = build_explicit_path(-1, forward_hops);
    }

    TT_FATAL(
        e_coord.has_value() || w_coord.has_value() || n_coord.has_value() || s_coord.has_value(),
        "No neighboring devices");

    const uint32_t packet_size = operation_attributes.packet_size;
    const bool receiver_l1_mode = use_receiver_l1_path(operation_attributes, tensor_args, output_tensor);
    const ReceiverL1StageMode receiver_stage_mode =
        receiver_l1_mode ? operation_attributes.receiver_policy.stage_mode : ReceiverL1StageMode::Combined;
    const bool receiver_send_payload = receiver_stage_mode != ReceiverL1StageMode::DrainOnly;
    const bool receiver_credit_enabled =
        receiver_stage_mode == ReceiverL1StageMode::Combined || receiver_stage_mode == ReceiverL1StageMode::L1Sink;
    const bool receiver_attribution = operation_attributes.receiver_policy.attribution_enabled;
    const bool receiver_address_attribution = operation_attributes.receiver_policy.address_attribution_enabled;
    const uint32_t receiver_cores_per_link = operation_attributes.receiver_policy.interleaved_bank_receivers ? 4 : 1;
    const uint32_t receiver_drain_risc_count =
        receiver_l1_mode ? receiver_l1_drain_risc_count(operation_attributes.receiver_policy, input_tensor) : 1;
    TT_FATAL(
        receiver_drain_risc_count == 1 ||
            operation_attributes.receiver_policy.credit_mode != ReceiverL1CreditMode::PerSlot,
        "dual-RISC receiver drain requires window or pipelined credits");
    TT_FATAL(
        receiver_drain_risc_count == 1 || receiver_stage_mode == ReceiverL1StageMode::Combined ||
            receiver_stage_mode == ReceiverL1StageMode::DrainOnly,
        "dual-RISC receiver drain is supported only for combined or drain_only stages");
    TT_FATAL(
        !operation_attributes.receiver_policy.interleaved_bank_receivers ||
            operation_attributes.receiver_policy.bank_owned_links,
        "interleaved bank receivers require bank-owned links");
    uint32_t receiver_slot_count = receiver_l1_mode ? operation_attributes.receiver_policy.slot_count : 1;
    TT_FATAL(
        !receiver_l1_mode || receiver_control_sems.size() ==
                                 num_devices +
                                     (operation_attributes.receiver_policy.bank_owned_links ? num_devices : 0) +
                                     2 * receiver_cores_per_link + (receiver_drain_risc_count == 2 ? 1 : 0),
        "receiver all-gather control semaphore count does not match the selected drain RISC count");

    // Kernel alternates between ranges[] and ranges_alt[] hops on every packet send.
    // Enabled if any axis is an even-sized ring.
    const bool load_balance_across_alt_routes = ew_load_balance || ns_load_balance;
    // Fresh outputs require the barrier so that every remote allocation exists before
    // traffic starts. Receiver-L1 mode also requires it for persistent outputs: the
    // producer must not publish into receiver slots until all receiver workers have
    // initialized their slot semaphores and begun consuming them.
    const bool do_init_barrier = receiver_l1_mode || !tensor_args.persistent_output_tensor.has_value();

    ////////////////////////////////////////////////////////////////
    // Core selection
    ////////////////////////////////////////////////////////////////

    // We allocate one worker core per link, but not per axis.
    // Each worker handles both dirs (forward and backward) and also both axes (N/S and E/W).
    // Known limitation: when the two axes have unequal link counts, the larger axis's extra links
    // go unused. If this is ever a real use-case, we need to allocate separate worker cores per axis.
    const uint32_t links0 = operation_attributes.axis_num_links[0];
    const uint32_t links1 = operation_attributes.axis_num_links[1];
    const uint32_t min_num_links = std::min(links0 > 0 ? links0 : links1, links1 > 0 ? links1 : links0);

    // Receiver mode adds one drain core per link.
    const uint32_t num_cores_per_link = receiver_l1_mode ? 1 + receiver_cores_per_link : 1;
    // On a four-device Blackhole gather axis, pairing FP8 sender/receiver cores
    // vertically avoids the NOC contention observed with the default horizontal
    // pairs.  The placement does not help the two-device Sparse MLA proxy, so
    // scope it to the measured four-participant path.  Keep explicit user
    // sub-grids in their established row-major order; topology-specific
    // placement for those grids is validated separately.
    const auto core_allocation_strategy = receiver_l1_mode && input_tensor.dtype() == ttnn::DataType::FP8_E4M3 &&
                                                  operation_attributes.num_devices == 4 &&
                                                  !operation_attributes.sub_core_grid.has_value()
                                              ? CoreAllocationStrategy::COL_MAJOR
                                              : CoreAllocationStrategy::ROW_MAJOR;
    auto [all_core_range, all_cores] = ttnn::ccl::choose_worker_cores(
        min_num_links,
        num_cores_per_link,
        input_tensor.device(),
        operation_attributes.subdevice_id,
        /*core_grid_offset=*/CoreCoord{0, 0},
        operation_attributes.sub_core_grid,
        core_allocation_strategy);
    TT_FATAL(
        all_cores.size() == static_cast<size_t>(min_num_links) * num_cores_per_link,
        "all_gather needs {} cores ({} links x {} cores/link) but only {} are available",
        static_cast<size_t>(min_num_links) * num_cores_per_link,
        min_num_links,
        num_cores_per_link,
        all_cores.size());

    std::vector<CoreCoord> worker_cores;
    std::vector<CoreCoord> receiver_cores;
    std::set<CoreRange> worker_core_set;
    std::set<CoreRange> receiver_core_set;
    for (uint32_t link = 0; link < min_num_links; ++link) {
        const auto& worker = all_cores[link * num_cores_per_link];
        worker_cores.push_back(worker);
        worker_core_set.emplace(worker, worker);
        if (receiver_l1_mode) {
            for (uint32_t receiver_idx = 0; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                const auto& receiver = all_cores[link * num_cores_per_link + 1 + receiver_idx];
                receiver_cores.push_back(receiver);
                receiver_core_set.emplace(receiver, receiver);
            }
        }
    }
    const CoreRangeSet worker_core_range(worker_core_set);
    const CoreRangeSet receiver_core_range(receiver_core_set);

    ////////////////////////////////////////////////////////////////
    // Page indexing
    //
    // Glossary:
    //   input page     -- one page of the input tensor.
    //   output page    -- one page of the output tensor (the real buffer page).
    //   chunk          -- one NOC write = min(input_page, output_page) bytes. An input
    //                     page = split_factor chunks; an output page = output_chunks_per_page
    //                     chunks. The kernel iterator walks chunks.
    //   stripe         -- a run of consecutive chunks this device writes before
    //                     jumping past other devices' contributions.
    //   stripe jump    -- value the kernel adds to output_page_id at the stripe
    //                     boundary.
    //
    // Three copy modes, picked by input vs output page sizes:
    //   matched (in == out): 1 chunk per input page, output_chunks_per_page = 1.
    //   concat  (out > in) : 1 chunk per input page, output_chunks_per_page > 1; each
    //                        chunk lands at a byte offset within a shared output page.
    //   split   (in > out) : split_factor chunks per input page, output_chunks_per_page = 1.
    //
    // Kernel is a dumb chunk iterator. Iteration pattern is:
    //   byte_offset++ within an output page -> chunk++ -> stripe+=jump
    //
    // Host derives the iterator parameters from input/output page sizes, gather dim,
    // and device index.
    ////////////////////////////////////////////////////////////////

    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();

    auto input_shape = input_tensor.padded_shape();
    uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }

    // --- Copy mode ---
    // The kernel always reads whole *aligned* input pages into L1 (required by the input's NoC
    // read alignment, DRAM or L1) but writes at output *content* (unaligned) granularity, so
    // chunk sizing differs by mode:
    //   matched (in == out): 1 chunk per input page, output_chunks_per_page = 1.
    //   concat  (out > in) : 1 chunk per input page, output_chunks_per_page > 1; each chunk
    //                        lands at a byte offset within a shared output page.
    //   split   (in > out) : split_factor chunks per input page, output_chunks_per_page = 1.
    const uint32_t input_unaligned_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_unaligned_page_size = output_tensor.buffer()->page_size();
    // matched/concat write a whole aligned input page (== L1 read stride) into an output slot;
    // split writes output-content-sized pieces to separate output page bases.
    const bool is_split = input_unaligned_page_size > output_unaligned_page_size;
    const uint32_t output_chunk_size = is_split ? output_unaligned_page_size : input_page_size;
    const uint32_t output_chunks_per_page = is_split ? 1u : output_unaligned_page_size / input_unaligned_page_size;
    const uint32_t split_factor = is_split ? input_unaligned_page_size / output_unaligned_page_size : 1u;
    TT_FATAL(
        output_chunks_per_page == 1 || input_page_size == input_unaligned_page_size,
        "concat requires an unpadded input page");  // so slots align to content

    const uint32_t total_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t batch_size = input_shape[0];
    TT_FATAL(
        total_input_pages % batch_size == 0,
        "all_gather input pages {} must divide evenly across batch size {}",
        total_input_pages,
        batch_size);
    const uint32_t pages_per_batch = total_input_pages / batch_size;
    const uint32_t input_batch_page_offset = operation_attributes.batch_slice_idx.value_or(0) * pages_per_batch;

    ::ttnn::ccl::validate_packet_size(input_tensor.device()->arch(), packet_size, output_chunk_size);

    // --- CB sizing ---
    // cb_page_size is a multiple of input_page_size, which is itself a multiple of
    // output_chunk_size = min(input, output), so the kernel increments both
    // the cb_read_ptr and cb_write_ptr cleanly.
    const uint32_t max_pages_per_packet = std::max(1u, packet_size / input_page_size);
    const bool bank_owned_links = operation_attributes.receiver_policy.bank_owned_links;
    const uint32_t bank_owned_coalesce = operation_attributes.receiver_policy.bank_owned_coalesce_mask;
    const uint32_t num_dram_banks = input_tensor.device()->num_dram_channels();
    TT_FATAL(bank_owned_links || bank_owned_coalesce == 0, "bank-owned coalescing requires the bank-owned schedule");
    uint32_t bank_owned_pages_per_run = 1;
    if (bank_owned_links) {
        TT_FATAL(receiver_l1_mode, "bank-owned links require the receiver-L1 all-gather path");
        TT_FATAL(
            receiver_stage_mode == ReceiverL1StageMode::Combined,
            "bank-owned links initially support only the combined receiver stage");
        TT_FATAL(
            input_tensor.device()->arch() == tt::ARCH::BLACKHOLE,
            "bank-owned links are initially validated only on Blackhole");
        TT_FATAL(
            num_devices >= 2 && num_devices <= 8, "bank-owned links support two to eight devices, got {}", num_devices);
        TT_FATAL(min_num_links == 2, "bank-owned links initially require two links, got {}", min_num_links);
        TT_FATAL(num_dram_banks == 8, "bank-owned links initially require eight DRAM banks, got {}", num_dram_banks);
        TT_FATAL(batch_size == 1, "bank-owned links initially require batch size one, got {}", batch_size);
        TT_FATAL(
            !operation_attributes.batch_slice_idx.has_value() && !operation_attributes.valid_gather_extent.has_value(),
            "bank-owned links do not yet support batch slices or partial gather extents");
        TT_FATAL(
            input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM &&
                output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "bank-owned links require DRAM input and output buffers");
        TT_FATAL(
            input_tensor.buffer()->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
                output_tensor.buffer()->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "bank-owned links require interleaved input and output buffers");
        TT_FATAL(
            total_input_pages % num_dram_banks == 0,
            "bank-owned links require {} input pages to divide across {} DRAM banks",
            total_input_pages,
            num_dram_banks);
        const uint32_t pages_per_bank = total_input_pages / num_dram_banks;
        bank_owned_pages_per_run = bank_owned_rows_per_run(
            operation_attributes.receiver_policy,
            receiver_l1_batch_rows(operation_attributes.receiver_policy, max_pages_per_packet),
            pages_per_bank);
        TT_FATAL(
            bank_owned_pages_per_run > 1,
            "bank-owned links need at least two contiguous pages per bank run; max rows {}, pages per bank {}",
            max_pages_per_packet,
            pages_per_bank);
    }
    const uint32_t pages_per_packet =
        bank_owned_links
            ? bank_owned_pages_per_run
            : (receiver_l1_mode ? receiver_l1_batch_rows(operation_attributes.receiver_policy, max_pages_per_packet)
                                : max_pages_per_packet);
    uint32_t cb_page_size = input_page_size * pages_per_packet;
    uint32_t cb_depth = 3;

    // Perf hack: for tile layout, pack multiple pages into a single CB page to reduce CB sync
    // frequency between reader and writer. Note this increases effective CB depth.
    // Don't do this for row-major layout because of all the careful handling of page sizes.
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        // Empirically determined heuristic, works well for all tensor sizes
        const uint32_t ideal_multiplier = (input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) ? 4 : 3;
        // Find the largest multiplier in [1, ideal] that fits in available L1
        const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
        const uint32_t multiplier = std::clamp(max_l1_space / (cb_depth * cb_page_size), 1u, ideal_multiplier);
        if (multiplier < ideal_multiplier) {
            log_warning(
                tt::LogOp,
                "CircularBuffer depth reduced due to L1 pressure (only {} B available), performance may regress.",
                max_l1_space);
        }
        cb_page_size *= multiplier;
    }

    TT_FATAL(
        !receiver_l1_mode || (output_chunks_per_page == 1 && split_factor == 1 && output_chunk_size == input_page_size),
        "receiver all-gather currently requires matched one-page row-major input/output geometry");
    const uint32_t receiver_buffer_base =
        input_tensor.device()->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    if (receiver_l1_mode && receiver_slot_count == 0) {
        const uint32_t available_receiver_l1 = input_tensor.device()->l1_size_per_core() - receiver_buffer_base;
        receiver_slot_count = available_receiver_l1 / (num_devices * cb_page_size);
        TT_FATAL(receiver_slot_count > 0, "receiver all-gather has no room for one complete source-slot set");
    }
    const bool receiver_proactive_credit =
        operation_attributes.receiver_policy.credit_mode == ReceiverL1CreditMode::Pipelined;
    uint32_t receiver_credit_group_batches = receiver_slot_count;
    if (operation_attributes.receiver_policy.credit_mode == ReceiverL1CreditMode::PerSlot) {
        receiver_credit_group_batches = 1;
    } else if (receiver_proactive_credit) {
        receiver_credit_group_batches = operation_attributes.receiver_policy.credit_group_batches == 0
                                            ? std::max(1u, receiver_slot_count / 2)
                                            : operation_attributes.receiver_policy.credit_group_batches;
    }
    TT_FATAL(
        !receiver_l1_mode ||
            (receiver_credit_group_batches > 0 && receiver_credit_group_batches <= receiver_slot_count),
        "receiver credit group of {} batches must fit in the {}-slot receiver ring",
        receiver_credit_group_batches,
        receiver_slot_count);
    const uint32_t receiver_credit_sem_base = 0;
    const uint32_t receiver_consumed_sem_base = receiver_cores_per_link;
    const uint32_t receiver_produced_forward_sem_base = 2 * receiver_cores_per_link;
    const uint32_t receiver_produced_backward_sem_base = receiver_produced_forward_sem_base + num_devices;
    const uint32_t receiver_dual_sync_sem_index =
        receiver_produced_backward_sem_base + (bank_owned_links ? num_devices : 0);
    const uint64_t receiver_buffer_end =
        receiver_buffer_base + static_cast<uint64_t>(num_devices) * receiver_slot_count * cb_page_size;
    TT_FATAL(
        !receiver_l1_mode || receiver_buffer_end <= input_tensor.device()->l1_size_per_core(),
        "receiver all-gather needs {} B of staging L1 from address {}, ending at {}, but core L1 ends at {}",
        static_cast<uint64_t>(num_devices) * receiver_slot_count * cb_page_size,
        receiver_buffer_base,
        receiver_buffer_end,
        input_tensor.device()->l1_size_per_core());

    // --- Stripe geometry ---
    // input_pages_per_stripe = num input pages along [gather_dim .. rank-1] this
    // device contributes per stripe. For RM gather_dim=-1 this is the *page* count,
    // which handles sharded RM input (> 1 input page per row).
    auto tile_spec = input_tensor.layout() == Layout::TILE ? input_tensor.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_full_stripe = 1;
    uint32_t input_pages_per_selected_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        const uint32_t full_extent = input_shape[i];
        const uint32_t selected_extent = (i == gather_dim && operation_attributes.valid_gather_extent.has_value())
                                             ? operation_attributes.valid_gather_extent.value()
                                             : full_extent;
        if (i == rank - 1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                input_pages_per_full_stripe *= full_extent / tile_spec.get_width();
                input_pages_per_selected_stripe *= selected_extent / tile_spec.get_width();
            } else {
                // This is a page count, so divide by the unaligned page size, not aligned
                input_pages_per_full_stripe *= (full_extent * input_tensor.element_size()) / input_unaligned_page_size;
                input_pages_per_selected_stripe *=
                    (selected_extent * input_tensor.element_size()) / input_unaligned_page_size;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == rank - 2) {
            input_pages_per_full_stripe *= full_extent / tile_spec.get_height();
            input_pages_per_selected_stripe *= selected_extent / tile_spec.get_height();
        } else {
            input_pages_per_full_stripe *= full_extent;
            input_pages_per_selected_stripe *= selected_extent;
        }
    }

    const bool partial_gather = operation_attributes.valid_gather_extent.has_value();
    const bool single_batch_gather = operation_attributes.batch_slice_idx.has_value();
    const uint32_t num_input_pages =
        partial_gather ? input_pages_per_selected_stripe : (single_batch_gather ? pages_per_batch : total_input_pages);
    TT_FATAL(
        (!partial_gather && !single_batch_gather) || num_input_pages <= pages_per_batch,
        "selected gather pages {} exceed pages per batch {}",
        num_input_pages,
        pages_per_batch);
    const uint32_t num_output_chunks = num_input_pages * split_factor;

    // Stripe = this device's contiguous run of chunks per row = input_pages_per_stripe
    // * split_factor. Measured in chunks (not output pages) so multi-shard concat works:
    // a stripe's chunks are laid across output pages via the inner byte-offset counter
    // and may straddle pages.
    // The output retains its full per-device slab. A partial gather writes only
    // the leading selected pages into each slab, preserving a fixed output layout.
    const uint32_t output_chunks_per_stripe = input_pages_per_full_stripe * split_factor;
    const uint32_t stripe_distance_chunks = num_devices * output_chunks_per_stripe;
    const uint32_t output_pages_per_row = stripe_distance_chunks / output_chunks_per_page;
    // This device's chunk phase within the output page. Constant across rows because
    // output_chunks_per_page divides stripe_distance_chunks (valid output sharding).
    const uint32_t off_start_chunks = (device_idx * output_chunks_per_stripe) % output_chunks_per_page;
    // Page carries accumulated while walking one full stripe.
    const uint32_t in_stripe_carries = (off_start_chunks + output_chunks_per_stripe - 1) / output_chunks_per_page;
    // Value added to output_page_id at the stripe boundary (jump to this device's run
    // in the next row): pages_per_row minus the carries already taken within the stripe.
    const uint32_t output_page_stripe_jump = output_pages_per_row - in_stripe_carries;
    // Per-device byte offset phase the iterator resets to at each stripe boundary.
    const uint32_t output_page_byte_offset = off_start_chunks * output_chunk_size;
    TT_FATAL(output_chunks_per_stripe > 0, "output_chunks_per_stripe must be > 0");
    if (bank_owned_links) {
        TT_FATAL(
            input_batch_page_offset % num_dram_banks == 0,
            "bank-owned input offset {} must preserve {}-bank alignment",
            input_batch_page_offset,
            num_dram_banks);
        TT_FATAL(
            input_pages_per_full_stripe == total_input_pages,
            "bank-owned links initially require one complete input stripe ({} pages versus {} total)",
            input_pages_per_full_stripe,
            total_input_pages);
        TT_FATAL(
            input_pages_per_full_stripe % num_dram_banks == 0,
            "bank-owned output source stride {} must preserve {}-bank alignment",
            input_pages_per_full_stripe,
            num_dram_banks);
        TT_FATAL(
            num_input_pages % min_num_links == 0,
            "bank-owned pages {} must divide evenly across {} links",
            num_input_pages,
            min_num_links);
        TT_FATAL(
            pages_per_packet > 0,
            "bank-owned run size must be positive for {} pages per bank",
            num_input_pages / num_dram_banks);
    }

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // Input CB
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);

    CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    // KERNEL CREATION
    // Reader (covers forward directions E-line + S-rect)
    const bool ring_fast_control_atomics =
        receiver_l1_mode && operation_attributes.receiver_policy.interleaved_bank_receivers &&
        operation_attributes.axis_topology[active_axis] == tt::tt_fabric::Topology::Ring;
    std::vector<uint32_t> reader_compile_args = {
        input_page_size,                 // input tensor page size
        output_chunk_size,               // NOC write size = min(input, output)
        output_chunks_per_page,          // chunks per output buffer page (1 unless concat)
        output_chunks_per_stripe,        // stripe length in chunks (before a stripe jump)
        output_page_stripe_jump,         // value added to output_page_id at stripe boundary
        cb0_id,                          // cb id
        cb_depth,                        // cb depth
        cb_page_size,                    // cb entry size
        packet_size,                     // packet_size
        load_balance_across_alt_routes,  // load_balance_across_alt_routes
        (e_hops > 0) + (s_hops > 0),     // num_connections
        do_init_barrier,                 // do_init_barrier
        receiver_l1_mode,                // remote payload terminates in receiver Tensix L1
        receiver_slot_count,             // remote L1 slots per source
        operation_attributes.receiver_policy.notify_mode ==
            ReceiverL1NotifyMode::Fused,  // payload plus produced notification
        receiver_credit_enabled,          // require consume/credit ordering
        operation_attributes.receiver_policy.credit_mode ==
            ReceiverL1CreditMode::Window,  // one acknowledgement per slot window
        receiver_proactive_credit,         // proxy completed credit groups before slot reuse
        receiver_credit_group_batches,     // batches represented by one consumed/credit sequence
        receiver_send_payload,             // suppress Fabric payloads for drain-only attribution
        receiver_attribution,              // emit accumulated per-stage profiler cycle records
        receiver_address_attribution,      // emit physical page-mapping counters
        bank_owned_links,                  // link owns complete DRAM-bank runs
        num_dram_banks,                    // interleaved DRAM-bank count
        min_num_links,                     // number of bank ownership groups
        bank_owned_coalesce,               // bit 0: source, bit 1: local output, bit 2: receiver
        receiver_cores_per_link,           // independent receiver/credit streams per link
        ring_fast_control_atomics,         // returned credits do not order terminal payload writes
        use_explicit_ring_path,            // physical route may turn while following the logical ring
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer (covers backward directions W-line + N-rect)
    std::vector<uint32_t> writer_compile_args = {
        output_chunk_size,               // NOC write size = min(input, output)
        output_chunks_per_page,          // chunks per output buffer page (1 unless concat)
        output_chunks_per_stripe,        // stripe length in chunks (before a stripe jump)
        output_page_stripe_jump,         // value added to output_page_id at stripe boundary
        cb0_id,                          // cb id
        cb_page_size,                    // cb entry size
        packet_size,                     // packet_size
        load_balance_across_alt_routes,  // load_balance_across_alt_routes
        (w_hops > 0) + (n_hops > 0),     // num_connections
        do_init_barrier,                 // do_init_barrier
        receiver_l1_mode,                // remote payload terminates in receiver Tensix L1
        receiver_slot_count,             // remote L1 slots per source
        operation_attributes.receiver_policy.notify_mode ==
            ReceiverL1NotifyMode::Fused,  // payload plus produced notification
        receiver_credit_enabled,          // require consume/credit ordering
        operation_attributes.receiver_policy.credit_mode ==
            ReceiverL1CreditMode::Window,  // one acknowledgement per slot window
        receiver_proactive_credit,         // proxy completed credit groups before slot reuse
        receiver_credit_group_batches,     // batches represented by one consumed/credit sequence
        receiver_send_payload,             // suppress Fabric payloads for drain-only attribution
        receiver_attribution,              // emit accumulated per-stage profiler cycle records
        bank_owned_links,                  // link owns complete DRAM-bank runs
        num_dram_banks,                    // interleaved DRAM-bank count
        min_num_links,                     // number of bank ownership groups
        bank_owned_coalesce,               // bit 0: source, bit 1: local output, bit 2: receiver
        receiver_cores_per_link,           // independent receiver/credit streams per link
        ring_fast_control_atomics,         // returned credits do not order terminal payload writes
        use_explicit_ring_path,            // physical route may turn while following the logical ring
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_reader.cpp",
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // Writer
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_writer.cpp",
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    tt::tt_metal::KernelHandle receiver_kernel_id{};
    tt::tt_metal::KernelHandle receiver_reader_kernel_id{};
    if (receiver_l1_mode) {
        std::vector<uint32_t> receiver_compile_args = {
            output_chunk_size,               // matched output page/write size
            cb_page_size,                    // one source slot stride
            cb_page_size / input_page_size,  // rows in a full transport batch
            num_devices,                     // source slots and produced semaphores
            receiver_buffer_base,            // raw dedicated-core staging base
            receiver_stage_mode == ReceiverL1StageMode::Combined ||
                receiver_stage_mode == ReceiverL1StageMode::DrainOnly,  // enable persistent-output drain
            receiver_slot_count,                                        // independent payload slots per source
            receiver_stage_mode != ReceiverL1StageMode::L1Overwrite,    // execute receiver batch schedule
            receiver_credit_group_batches,                              // batches represented by one consumed sequence
            receiver_send_payload,         // wait for produced and publish consumption when payloads are sent
            receiver_attribution,          // emit accumulated per-stage profiler cycle records
            receiver_drain_risc_count,     // one or both data-movement RISCs drain receiver slots
            0,                             // BRISC receiver-drain role
            receiver_address_attribution,  // emit physical page-mapping counters
            bank_owned_links,              // link owns complete DRAM-bank runs
            num_dram_banks,                // interleaved DRAM-bank count
            min_num_links,                 // number of bank ownership groups
            bank_owned_coalesce,           // bit 0: source, bit 1: local output, bit 2: receiver
            receiver_cores_per_link,       // receiver cores per active Fabric link
            operation_attributes.axis_topology[operation_attributes.axis_num_devices[0] > 1 ? 0 : 1] ==
                tt::tt_fabric::Topology::Ring,  // active axis wraps
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(receiver_compile_args);
        receiver_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_receiver_writer.cpp",
            receiver_core_range,
            tt::tt_metal::WriterDataMovementConfig(receiver_compile_args));
        if (receiver_drain_risc_count == 2) {
            auto receiver_reader_compile_args = receiver_compile_args;
            receiver_reader_compile_args[12] = 1;  // NCRISC receiver-drain role
            receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_receiver_writer.cpp",
                receiver_core_range,
                tt::tt_metal::ReaderDataMovementConfig(receiver_reader_compile_args));
        }
    }

    ////////////////////////////////////////////////////////////////
    // Runtime args
    ////////////////////////////////////////////////////////////////

    for (uint32_t link = 0; link < min_num_links; link++) {
        CoreCoord core = worker_cores[link];
        CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
        CoreCoord receiver_core{};
        CoreCoord virtual_receiver_core{};
        std::vector<CoreCoord> virtual_receiver_cores;
        if (receiver_l1_mode) {
            receiver_core = receiver_cores[link * receiver_cores_per_link];
            virtual_receiver_core = mesh_device->worker_core_from_logical_core(receiver_core);
            for (uint32_t receiver_idx = 0; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                virtual_receiver_cores.push_back(mesh_device->worker_core_from_logical_core(
                    receiver_cores[link * receiver_cores_per_link + receiver_idx]));
            }
        }

        // Set runtime args
        uint32_t input_pages_per_link = num_input_pages / min_num_links;
        uint32_t remainder = num_input_pages % min_num_links;
        uint32_t selected_input_page_start = (link * input_pages_per_link) + std::min(link, remainder);
        uint32_t selected_input_page_end = ((link + 1) * input_pages_per_link) + std::min(link + 1, remainder);
        uint32_t input_tile_id_start =
            bank_owned_links ? input_batch_page_offset : input_batch_page_offset + selected_input_page_start;
        uint32_t input_tile_id_end = bank_owned_links ? input_batch_page_offset + num_input_pages
                                                      : input_batch_page_offset + selected_input_page_end;

        // Map this worker's slice of input pages to its slice of output chunks.
        // num_output_chunks already accounts for split_factor, so in matched/concat
        // modes the ratio cancels back to num_input_pages.
        uint32_t local_output_start =
            (static_cast<uint64_t>(selected_input_page_start) * num_output_chunks) / num_input_pages;
        uint32_t local_output_end =
            (static_cast<uint64_t>(selected_input_page_end) * num_output_chunks) / num_input_pages;
        uint32_t num_worker_output_chunks = local_output_end - local_output_start;
        // s_start = global chunk index of this worker's first write:
        //     stripe_index  = local / output_chunks_per_stripe
        //     pos_in_stripe = local % output_chunks_per_stripe
        //     s_start       = stripe_index * stripe_distance_chunks    (skip other devices' rows)
        //                   + device_idx   * output_chunks_per_stripe  (this device's run in the row)
        //                   + pos_in_stripe
        uint32_t s_start = (local_output_start / output_chunks_per_stripe) * stripe_distance_chunks +
                           device_idx * output_chunks_per_stripe + local_output_start % output_chunks_per_stripe;
        uint32_t output_page_id_start = s_start / output_chunks_per_page;
        uint32_t output_page_byte_offset_start = (s_start % output_chunks_per_page) * output_chunk_size;
        uint32_t output_chunk_in_stripe_start = local_output_start % output_chunks_per_stripe;

        // Per-link barrier fan-in = N-1 in every case. Every other chip sends me one atomic_inc:
        //   1D: e_hops + w_hops (or n_hops + s_hops) along the active axis = axis_size - 1.
        //   2D: every chip in the mesh outside me is covered by exactly one of the 4 mcast packets.
        // Both equal num_devices - 1.
        uint32_t barrier_wait_value = num_devices - 1 + (receiver_l1_mode && receiver_cores_per_link == 1 ? 2 : 0);
        const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
        auto physical_direction = [&](const std::optional<MeshCoordinate>& coord) {
            if (!coord.has_value()) {
                return 0u;
            }
            return static_cast<uint32_t>(tt::tt_fabric::get_eth_forwarding_direction(
                                             sender_fabric_node_id, mesh_device->get_fabric_node_id(*coord))
                                             .value());
        };

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),   // input tensor address
            output_tensor.buffer()->address(),  // output tensor address
            input_tile_id_start,                // input_page_id_start
            input_tile_id_end,                  // input_page_id_end
            output_page_id_start,               // output page start
            output_chunk_in_stripe_start,       // initial chunk position within stripe
            output_page_byte_offset,            // per-device offset phase (reset at stripe boundary)
            output_page_byte_offset_start,      // worker's initial byte offset within output page
            num_worker_output_chunks,           // number of output chunks for this worker
            device_idx,                         // this device's index
            barrier_sem.address(),              // barrier_sem L1 address
            virtual_core.x,                     // barrier_sem location (core.x)
            virtual_core.y,                     // barrier_sem location (core.y)
            barrier_wait_value,                 // barrier counter to wait for
            e_hops,                             // line_hops
            e_hops,                             // rect_e_hops
            w_hops,                             // rect_w_hops
            s_hops,                             // rect_spine_hops
            ew_load_balance ? w_hops : e_hops,  // line_hops_alt
            ew_load_balance ? w_hops : e_hops,  // rect_e_hops_alt
            ew_load_balance ? e_hops : w_hops,  // rect_w_hops_alt
            ns_load_balance ? n_hops : s_hops,  // rect_spine_hops_alt
            physical_direction(e_coord),        // physical direction of the E line
            physical_direction(e_coord),        // physical direction of the E rectangle edge
            physical_direction(w_coord),        // physical direction of the W rectangle edge
            physical_direction(s_coord),        // physical direction of the S rectangle spine
            reader_path.length | (reader_path.escape_hop << 8),
            reader_path_alt.length | (reader_path_alt.escape_hop << 8),
            reader_path.words[0],
            reader_path.words[1],
            reader_path.words[2],
            reader_path.words[3],
            reader_path.words[4],
            reader_path_alt.words[0],
            reader_path_alt.words[1],
            reader_path_alt.words[2],
            reader_path_alt.words[3],
            reader_path_alt.words[4],
            virtual_receiver_core.x,  // receiver L1 core x (0 outside receiver mode)
            virtual_receiver_core.y,  // receiver L1 core y
            receiver_buffer_base,     // source-indexed receiver slot base
            receiver_l1_mode ? receiver_control_sems[receiver_produced_forward_sem_base + device_idx].address() : 0,
            receiver_l1_mode ? receiver_control_sems[receiver_credit_sem_base].address() : 0,
            receiver_l1_mode ? receiver_control_sems[receiver_consumed_sem_base].address() : 0,
            receiver_l1_mode ? num_devices - 1 : 0,
            link,  // bank-ownership link index
        };
        if (receiver_l1_mode) {
            for (uint32_t receiver_idx = 1; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                reader_rt_args.push_back(receiver_control_sems[receiver_credit_sem_base + receiver_idx].address());
                reader_rt_args.push_back(receiver_control_sems[receiver_consumed_sem_base + receiver_idx].address());
                reader_rt_args.push_back(virtual_receiver_cores[receiver_idx].x);
                reader_rt_args.push_back(virtual_receiver_cores[receiver_idx].y);
            }
        }
        // Reader forward connection info: E-line (axis 1) then S-rect (axis 0).
        std::vector<tt::tt_fabric::FabricNodeId> reader_dst_nodes;
        if (e_hops > 0 && e_coord.has_value()) {
            reader_dst_nodes.push_back(mesh_device->get_fabric_node_id(*e_coord));
        }
        if (s_hops > 0 && s_coord.has_value()) {
            reader_dst_nodes.push_back(mesh_device->get_fabric_node_id(*s_coord));
        }
        if (!reader_dst_nodes.empty()) {
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                reader_dst_nodes,
                {link},
                program,
                reader_kernel_id,
                {core},
                reader_rt_args,
                fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
        }

        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),  // output tensor address
            output_page_id_start,               // output page start
            output_chunk_in_stripe_start,       // initial chunk position within stripe
            output_page_byte_offset,            // per-device offset phase (reset at stripe boundary)
            output_page_byte_offset_start,      // worker's initial byte offset within output page
            num_worker_output_chunks,           // number of output chunks for this worker
            device_idx,                         // this device's index
            barrier_sem.address(),              // barrier_sem L1 address
            virtual_core.x,                     // barrier_sem location (core.x)
            virtual_core.y,                     // barrier_sem location (core.y)
            w_hops,                             // line_hops
            e_hops,                             // rect_e_hops
            w_hops,                             // rect_w_hops
            n_hops,                             // rect_spine_hops
            ew_load_balance ? e_hops : w_hops,  // line_hops_alt
            ew_load_balance ? w_hops : e_hops,  // rect_e_hops_alt
            ew_load_balance ? e_hops : w_hops,  // rect_w_hops_alt
            ns_load_balance ? s_hops : n_hops,  // rect_spine_hops_alt
            physical_direction(w_coord),        // physical direction of the W line
            physical_direction(e_coord),        // physical direction of the E rectangle edge
            physical_direction(w_coord),        // physical direction of the W rectangle edge
            physical_direction(n_coord),        // physical direction of the N rectangle spine
            writer_path.length | (writer_path.escape_hop << 8),
            writer_path_alt.length | (writer_path_alt.escape_hop << 8),
            writer_path.words[0],
            writer_path.words[1],
            writer_path.words[2],
            writer_path.words[3],
            writer_path.words[4],
            writer_path_alt.words[0],
            writer_path_alt.words[1],
            writer_path_alt.words[2],
            writer_path_alt.words[3],
            writer_path_alt.words[4],
            virtual_receiver_core.x,  // receiver L1 core x (0 outside receiver mode)
            virtual_receiver_core.y,  // receiver L1 core y
            receiver_buffer_base,     // source-indexed receiver slot base
            receiver_l1_mode
                ? receiver_control_sems
                      [(bank_owned_links ? receiver_produced_backward_sem_base : receiver_produced_forward_sem_base) +
                       device_idx]
                          .address()
                : 0,
            receiver_l1_mode ? receiver_control_sems[receiver_credit_sem_base].address() : 0,
            receiver_l1_mode ? receiver_control_sems[receiver_consumed_sem_base].address() : 0,
            receiver_l1_mode ? num_devices - 1 : 0,
            link,                         // bank-ownership link index
            input_pages_per_full_stripe,  // output page stride between source slabs
        };
        if (receiver_l1_mode) {
            for (uint32_t receiver_idx = 1; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                writer_rt_args.push_back(receiver_control_sems[receiver_credit_sem_base + receiver_idx].address());
                writer_rt_args.push_back(receiver_control_sems[receiver_consumed_sem_base + receiver_idx].address());
                writer_rt_args.push_back(virtual_receiver_cores[receiver_idx].x);
                writer_rt_args.push_back(virtual_receiver_cores[receiver_idx].y);
            }
        }

        // Writer backward connections: W-line (axis 1) then N-rect (axis 0).
        std::vector<tt::tt_fabric::FabricNodeId> writer_dst_nodes;
        if (w_hops > 0 && w_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*w_coord));
        }
        if (n_hops > 0 && n_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*n_coord));
        }
        if (!writer_dst_nodes.empty()) {
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                writer_dst_nodes,
                {link},
                program,
                writer_kernel_id,
                {core},
                writer_rt_args,
                fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
        }

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);

        if (receiver_l1_mode) {
            for (uint32_t receiver_idx = 0; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                const auto& indexed_receiver_core = receiver_cores[link * receiver_cores_per_link + receiver_idx];
                std::vector<uint32_t> receiver_rt_args = {
                    output_tensor.buffer()->address(),                 // persistent output address
                    device_idx,                                        // local source is already written by sender core
                    bank_owned_links ? 0 : selected_input_page_start,  // logical slice start or owned-page count base
                    bank_owned_links ? input_pages_per_link : selected_input_page_end,
                    input_pages_per_full_stripe,  // full output-page stride between source slabs
                    receiver_control_sems[receiver_consumed_sem_base + receiver_idx]
                        .address(),         // this receiver's consumed sequence on mirrored sender core
                    virtual_core.x,         // mirrored sender core x
                    virtual_core.y,         // mirrored sender core y
                    barrier_sem.address(),  // receiver-ready signal target
                    link,                   // bank-ownership link index
                    receiver_idx,           // bank slot owned by this receiver core
                };
                for (uint32_t source = 0; source < num_devices; ++source) {
                    receiver_rt_args.push_back(
                        receiver_control_sems[receiver_produced_forward_sem_base + source].address());
                }
                for (uint32_t source = 0; source < num_devices; ++source) {
                    receiver_rt_args.push_back(receiver_control_sems
                                                   [(bank_owned_links ? receiver_produced_backward_sem_base
                                                                      : receiver_produced_forward_sem_base) +
                                                    source]
                                                       .address());
                }
                receiver_rt_args.push_back(
                    receiver_drain_risc_count == 2 ? receiver_control_sems[receiver_dual_sync_sem_index].address() : 0);
                tt::tt_metal::SetRuntimeArgs(program, receiver_kernel_id, {indexed_receiver_core}, receiver_rt_args);
                if (receiver_drain_risc_count == 2) {
                    tt::tt_metal::SetRuntimeArgs(
                        program, receiver_reader_kernel_id, {indexed_receiver_core}, receiver_rt_args);
                }
            }
        }
    }

    shared_variables_t shared_variables{
        .worker_cores = worker_cores,
        .receiver_cores = receiver_cores,
        .reader_kernel_id = reader_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .receiver_kernel_id = receiver_kernel_id,
        .receiver_reader_kernel_id = receiver_reader_kernel_id,
        .receiver_drain_risc_count = receiver_l1_mode ? receiver_drain_risc_count : 0,
        .receiver_cores_per_link = receiver_cores_per_link,
        .bank_owned_links = bank_owned_links,
        .barrier_sem = barrier_sem,
        .receiver_control_sems = receiver_control_sems,
    };

    return {std::move(program), std::move(shared_variables)};
}

void AllGatherMulticastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& operation_attributes,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    auto input_shape = input_tensor.padded_shape();
    const uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t input_unaligned_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_unaligned_page_size = output_tensor.buffer()->page_size();
    const bool is_split = input_unaligned_page_size > output_unaligned_page_size;
    const uint32_t output_chunks_per_page = is_split ? 1u : output_unaligned_page_size / input_unaligned_page_size;
    const uint32_t split_factor = is_split ? input_unaligned_page_size / output_unaligned_page_size : 1u;

    const uint32_t total_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t batch_size = input_shape[0];
    const uint32_t pages_per_batch = total_input_pages / batch_size;
    const uint32_t input_batch_page_offset = operation_attributes.batch_slice_idx.value_or(0) * pages_per_batch;

    auto tile_spec = input_tensor.layout() == Layout::TILE ? input_tensor.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_full_stripe = 1;
    uint32_t input_pages_per_selected_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        const uint32_t full_extent = input_shape[i];
        const uint32_t selected_extent = (i == gather_dim && operation_attributes.valid_gather_extent.has_value())
                                             ? operation_attributes.valid_gather_extent.value()
                                             : full_extent;
        if (i == rank - 1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                input_pages_per_full_stripe *= full_extent / tile_spec.get_width();
                input_pages_per_selected_stripe *= selected_extent / tile_spec.get_width();
            } else {
                input_pages_per_full_stripe *= (full_extent * input_tensor.element_size()) / input_unaligned_page_size;
                input_pages_per_selected_stripe *=
                    (selected_extent * input_tensor.element_size()) / input_unaligned_page_size;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == rank - 2) {
            input_pages_per_full_stripe *= full_extent / tile_spec.get_height();
            input_pages_per_selected_stripe *= selected_extent / tile_spec.get_height();
        } else {
            input_pages_per_full_stripe *= full_extent;
            input_pages_per_selected_stripe *= selected_extent;
        }
    }
    const uint32_t num_input_pages =
        operation_attributes.valid_gather_extent.has_value()
            ? input_pages_per_selected_stripe
            : (operation_attributes.batch_slice_idx.has_value() ? pages_per_batch : total_input_pages);
    const uint32_t num_output_chunks = num_input_pages * split_factor;
    const uint32_t output_chunks_per_stripe = input_pages_per_full_stripe * split_factor;
    const uint32_t stripe_distance_chunks = operation_attributes.num_devices * output_chunks_per_stripe;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        const uint32_t barrier_sem_addr = shared_vars.barrier_sem.address();

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);
        auto* receiver_args_by_core =
            shared_vars.receiver_cores.empty() ? nullptr : &GetRuntimeArgs(program, shared_vars.receiver_kernel_id);
        auto* receiver_reader_args_by_core = shared_vars.receiver_drain_risc_count != 2
                                                 ? nullptr
                                                 : &GetRuntimeArgs(program, shared_vars.receiver_reader_kernel_id);
        for (uint32_t link = 0; link < shared_vars.worker_cores.size(); ++link) {
            const auto& core = shared_vars.worker_cores[link];
            // reader: [0]=input_addr, [1]=output_addr, [10]=barrier_sem
            auto& reader_args = reader_args_by_core[core.x][core.y];
            reader_args[0] = input_addr;
            reader_args[1] = output_addr;
            reader_args[10] = barrier_sem_addr;
            const uint32_t input_pages_per_link = num_input_pages / shared_vars.worker_cores.size();
            const uint32_t remainder = num_input_pages % shared_vars.worker_cores.size();
            const uint32_t selected_input_page_start = (link * input_pages_per_link) + std::min(link, remainder);
            const uint32_t selected_input_page_end =
                ((link + 1) * input_pages_per_link) + std::min(link + 1, remainder);
            const uint32_t local_output_start =
                (static_cast<uint64_t>(selected_input_page_start) * num_output_chunks) / num_input_pages;
            const uint32_t local_output_end =
                (static_cast<uint64_t>(selected_input_page_end) * num_output_chunks) / num_input_pages;
            const uint32_t num_worker_output_chunks = local_output_end - local_output_start;
            const uint32_t device_idx = reader_args[9];
            const uint32_t s_start = (local_output_start / output_chunks_per_stripe) * stripe_distance_chunks +
                                     device_idx * output_chunks_per_stripe +
                                     local_output_start % output_chunks_per_stripe;
            const uint32_t output_page_id_start = s_start / output_chunks_per_page;
            const uint32_t output_page_byte_offset_start =
                (s_start % output_chunks_per_page) * (is_split ? output_unaligned_page_size : input_page_size);
            const uint32_t output_chunk_in_stripe_start = local_output_start % output_chunks_per_stripe;
            reader_args[2] = shared_vars.bank_owned_links ? input_batch_page_offset
                                                          : input_batch_page_offset + selected_input_page_start;
            reader_args[3] = shared_vars.bank_owned_links ? input_batch_page_offset + num_input_pages
                                                          : input_batch_page_offset + selected_input_page_end;
            reader_args[4] = output_page_id_start;
            reader_args[5] = output_chunk_in_stripe_start;
            reader_args[7] = output_page_byte_offset_start;
            reader_args[8] = num_worker_output_chunks;
            // writer: [0]=output_addr, [7]=barrier_sem
            auto& writer_args = writer_args_by_core[core.x][core.y];
            writer_args[0] = output_addr;
            writer_args[7] = barrier_sem_addr;
            writer_args[1] = output_page_id_start;
            writer_args[2] = output_chunk_in_stripe_start;
            writer_args[4] = output_page_byte_offset_start;
            writer_args[5] = num_worker_output_chunks;
            if (receiver_args_by_core != nullptr) {
                for (uint32_t receiver_idx = 0; receiver_idx < shared_vars.receiver_cores_per_link; ++receiver_idx) {
                    const auto& receiver_core =
                        shared_vars.receiver_cores[link * shared_vars.receiver_cores_per_link + receiver_idx];
                    auto& receiver_args = (*receiver_args_by_core)[receiver_core.x][receiver_core.y];
                    receiver_args[0] = output_addr;
                    receiver_args[2] = shared_vars.bank_owned_links ? 0 : selected_input_page_start;
                    receiver_args[3] = shared_vars.bank_owned_links ? num_input_pages / shared_vars.worker_cores.size()
                                                                    : selected_input_page_end;
                    receiver_args[4] = input_pages_per_full_stripe;
                    receiver_args[8] = barrier_sem_addr;
                    if (receiver_reader_args_by_core != nullptr) {
                        auto& receiver_reader_args = (*receiver_reader_args_by_core)[receiver_core.x][receiver_core.y];
                        receiver_reader_args[0] = output_addr;
                        receiver_reader_args[2] = shared_vars.bank_owned_links ? 0 : selected_input_page_start;
                        receiver_reader_args[3] = shared_vars.bank_owned_links
                                                      ? num_input_pages / shared_vars.worker_cores.size()
                                                      : selected_input_page_end;
                        receiver_reader_args[4] = input_pages_per_full_stripe;
                        receiver_reader_args[8] = barrier_sem_addr;
                    }
                }
            }
        }
    }
}

}  // namespace ttnn::operations::ccl
