// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/ccl_common.hpp"

#include <cstdint>
#include <cmath>

#include "ccl_host_datastructures.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"

#include <tt-metalium/cluster.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "tt-metalium/hal.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::ccl {

bool is_fabric_2d() {
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();

    return fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D;
}

tt::tt_fabric::Topology convert_2d_to_1d_topology(tt::tt_fabric::Topology topology) {
    if (topology == tt::tt_fabric::Topology::Mesh || topology == tt::tt_fabric::Topology::Linear) {
        return tt::tt_fabric::Topology::Linear;
    }
    if (topology == tt::tt_fabric::Topology::Torus || topology == tt::tt_fabric::Topology::Ring) {
        return tt::tt_fabric::Topology::Ring;
    }
    return topology;
}

tt::tt_metal::distributed::MeshCoordinate::BoundaryMode get_boundary_mode(
    const Tensor& tensor, tt::tt_fabric::Topology topology, std::optional<uint32_t> cluster_axis) {
    auto mesh_shape = tensor.device()->shape();
    auto device_coords = tensor.device_storage().get_coords();
    TT_FATAL(!device_coords.empty(), "device_coords is empty");
    if (topology == tt::tt_fabric::Topology::Linear || topology == tt::tt_fabric::Topology::Mesh) {
        return tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
    }
    // ring is possible if device coordinates along our cluster axis are the same as the last coordinate in the mesh
    // shape first_index = 0 last index = mesh_shape[cluster_axis] - 1
    if (cluster_axis.has_value()) {
        if (mesh_shape[cluster_axis.value()] == 2) {
            return tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
        }
        bool first_index_is_0 = device_coords[0][cluster_axis.value()] == 0;
        bool last_index_is_mesh_shape_minus_1 =
            device_coords[device_coords.size() - 1][cluster_axis.value()] == mesh_shape[cluster_axis.value()] - 1;
        if (first_index_is_0 && last_index_is_mesh_shape_minus_1) {
            return tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP;
        }
        return tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
    }
    if (mesh_shape[0] == 2 || mesh_shape[1] == 2) {
        return tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
    }
    TT_FATAL(!device_coords.empty(), "device_coords is empty");
    for (int i = 0; i < device_coords.front().dims(); i++) {
        if (device_coords.front()[i] != 0) {
            return tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
        }
    }
    for (int i = 0; i < device_coords.back().dims(); i++) {
        if (device_coords.back()[i] != mesh_shape[i] - 1) {
            return tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
        }
    }

    return tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP;
}

tt::tt_fabric::Topology get_usable_topology(
    const Tensor& tensor,
    const std::optional<tt::tt_fabric::Topology>& topology,
    const std::optional<uint32_t>& cluster_axis) {
    tt::tt_fabric::Topology topology_ = topology.value_or(tt::tt_fabric::get_fabric_topology());
    if (topology_ == tt::tt_fabric::Topology::Ring || topology_ == tt::tt_fabric::Topology::Torus) {
        auto boundary_mode = get_boundary_mode(tensor, topology_, cluster_axis);
        if (boundary_mode == tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP) {
            return topology_;
        }
        if (topology_ == tt::tt_fabric::Topology::Torus) {
            return tt::tt_fabric::Topology::Mesh;
        }
        return tt::tt_fabric::Topology::Linear;
    }
    return topology_;
}

uint32_t get_topological_dimension(const Tensor& tensor, const std::optional<uint32_t>& cluster_axis) {
    const auto device_coords = tensor.device_storage().get_coords();
    TT_FATAL(!device_coords.empty(), "device_coords is empty");
    if (cluster_axis.has_value()) {
        log_debug(tt::LogOp, "Cluster axis has value {}", cluster_axis.value());
        TT_FATAL(!device_coords.empty(), "device_coords is empty");
        TT_FATAL(
            device_coords[0].dims() > cluster_axis.value(),
            "cluster axis {} is out of range for device coords rank {} ",
            cluster_axis.value(),
            device_coords[0].dims());
        uint32_t ring_size = 0;
        for (const auto& device_coord : device_coords) {
            ring_size = std::max(ring_size, device_coord[cluster_axis.value()] + 1);
        }
        TT_FATAL(ring_size > 0, "ring_size is 0");
        log_debug(tt::LogOp, "Topological dimension {}", ring_size);
        return ring_size;
    }
    log_debug(tt::LogOp, "Topological dimension {}", device_coords.size());
    return device_coords.size();
}

uint32_t get_linearized_index_from_physical_coord(
    const Tensor& tensor, const MeshCoordinate& physical_coord, const std::optional<uint32_t>& cluster_axis) {
    const auto device_coords = tensor.device_storage().get_coords();
    TT_FATAL(!device_coords.empty(), "device_coords is empty");
    if (cluster_axis.has_value()) {
        log_debug(tt::LogOp, "Cluster axis has value {}", cluster_axis.value());
        TT_FATAL(
            physical_coord.dims() > cluster_axis.value(),
            "cluster axis {} is out of range for physical coord rank {} ",
            cluster_axis.value(),
            physical_coord.dims());
        // find minimum value along the cluster axis
        uint32_t min_value = std::numeric_limits<uint32_t>::max();
        for (const auto& device_coord : device_coords) {
            min_value = std::min(min_value, device_coord[cluster_axis.value()]);
        }
        TT_FATAL(
            physical_coord[cluster_axis.value()] >= min_value,
            "physical_coord[{}] {} is less than min_value {}",
            cluster_axis.value(),
            physical_coord[cluster_axis.value()],
            min_value);
        log_debug(
            tt::LogOp,
            "Physical linearized index for physical_coord: {} is {}",
            physical_coord,
            physical_coord[cluster_axis.value()] - min_value);
        return physical_coord[cluster_axis.value()] - min_value;
    }
    auto it = std::find(device_coords.begin(), device_coords.end(), physical_coord);
    TT_FATAL(it != device_coords.end(), "physical_coord not found in device_coords");
    log_debug(
        tt::LogOp,
        "Physical linearized index for physical_coord: {} is {}",
        physical_coord,
        static_cast<uint32_t>(std::distance(device_coords.begin(), it)));
    return static_cast<uint32_t>(std::distance(device_coords.begin(), it));
}

std::optional<MeshCoordinate> get_physical_neighbor_from_physical_coord(
    const Tensor& tensor,
    const MeshCoordinate& physical_coord,
    int offset,
    ttnn::ccl::Topology topology,
    const std::optional<uint32_t>& cluster_axis) {
    const auto device_coords = tensor.device_storage().get_coords();
    TT_FATAL(!device_coords.empty(), "device_coords is empty");
    auto boundary_mode = get_boundary_mode(tensor, topology, cluster_axis);
    if (cluster_axis.has_value()) {
        TT_FATAL(
            device_coords[0][cluster_axis.value()] == 0,
            "Currently, we only support CCLs with physical coordinates starting from 0 along the cluster axis {}, we "
            "got {}",
            cluster_axis.value(),
            device_coords[0][cluster_axis.value()]);
        TT_FATAL(
            physical_coord.dims() > cluster_axis.value(),
            "cluster axis {} is out of range for physical coord rank {} ",
            cluster_axis.value(),
            physical_coord.dims());
        log_debug(tt::LogOp, "Boundary mode: {}", boundary_mode);
        auto potential_neighbor =
            physical_coord.get_neighbor(tensor.device()->shape(), offset, cluster_axis.value(), boundary_mode);
        auto it = std::find(device_coords.begin(), device_coords.end(), potential_neighbor);
        if (it != device_coords.end()) {
            log_debug(
                tt::LogOp,
                "Physical coord {} Potential neighbor {} is found in device_coords",
                physical_coord,
                potential_neighbor);
            return potential_neighbor;
        }
        log_debug(
            tt::LogOp,
            "Physical coord {} Potential neighbor {} is not found in device_coords",
            physical_coord,
            potential_neighbor);
        return std::nullopt;
    }
    uint32_t physical_linearized_index = get_linearized_index_from_physical_coord(tensor, physical_coord, cluster_axis);
    int potential_neighbor_idx = (int)physical_linearized_index + offset;
    if (boundary_mode == tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP) {
        potential_neighbor_idx = (potential_neighbor_idx + device_coords.size()) % device_coords.size();
    } else if (potential_neighbor_idx < 0 || potential_neighbor_idx >= static_cast<int>(device_coords.size())) {
        log_debug(
            tt::LogOp,
            "Potential neighbor idx {} is out of range for device_coords size {}",
            potential_neighbor_idx,
            device_coords.size());
        return std::nullopt;
    }
    log_debug(tt::LogOp, "Potential neighbor idx {} is found in device_coords", potential_neighbor_idx);
    return device_coords[potential_neighbor_idx];
}

void SyncModeSpec::add_signal(uint32_t sem_id, uint32_t wait_count) {
    this->sem_ids.push_back(sem_id);
    this->wait_counts.push_back(wait_count);
    this->num_signals++;
}

LineTopology::LineTopology(size_t line_size, size_t line_index) : _line_size(line_size), _line_index(line_index) {}

bool LineTopology::is_first_device_in_line(ttnn::ccl::LineDirection direction) const {
    if (direction == ttnn::ccl::LineDirection::FORWARD) {
        return _line_index == 0;
    }
    TT_ASSERT(direction == ttnn::ccl::LineDirection::BACKWARD);
    return _line_index == _line_size - 1;
}
bool LineTopology::is_last_device_in_line(ttnn::ccl::LineDirection direction) const {
    if (direction == ttnn::ccl::LineDirection::BACKWARD) {
        return _line_index == 0;
    }
    TT_ASSERT(direction == ttnn::ccl::LineDirection::FORWARD);
    return _line_index == _line_size - 1;
}

bool LineTopology::is_at_end_of_line() const { return _line_index == 0 || _line_index == _line_size - 1; }

size_t LineTopology::line_size() const { return _line_size; }

size_t LineTopology::line_index() const { return _line_index; }

size_t LineTopology::get_distance_to_end_of_line(ttnn::ccl::LineDirection direction) const {
    if (direction == ttnn::ccl::LineDirection::FORWARD) {
        return (_line_size - _line_index) - 1;
    }
    return _line_index;
}

ttnn::ccl::Topology LineTopology::topology() const { return ttnn::ccl::Topology::Linear; }

SenderReceiverConfig get_device_sender_receiver_config(
    const IDevice* target_device, const std::vector<IDevice*>& devices, ttnn::ccl::Topology topology) {
    uint32_t num_devices = devices.size();
    bool is_linear = topology == ttnn::ccl::Topology::Linear;
    SenderReceiverConfig config;
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == target_device) {
            config.device_index = i;
            bool is_last_chip_in_clockwise_direction = is_linear && i == (num_devices - 1);
            bool is_last_chip_in_counter_clockwise_direction = is_linear && i == 0;

            config.receiver_device_id = is_last_chip_in_clockwise_direction
                                            ? std::nullopt
                                            : std::optional<tt::ChipId>(devices.at((i + 1) % num_devices)->id());

            config.sender_device_id =
                is_last_chip_in_counter_clockwise_direction
                    ? std::nullopt
                    : std::optional<tt::ChipId>(devices.at((i + num_devices - 1) % num_devices)->id());
        }
    }

    return config;
}

SenderReceiverConfig get_device_sender_receiver_config_in_ring(
    const MeshCoordinate& mesh_coord,
    const distributed::MeshDevice* mesh_device,
    uint32_t cluster_axis,
    int ring_size) {
    SenderReceiverConfig config;
    const auto& mesh_view = mesh_device->get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "CLL operation invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    config.device_index = (cluster_axis == 0) ? mesh_coord[0] : mesh_coord[1];

    auto get_chip_id = [&](std::size_t line_index) -> std::optional<tt::ChipId> {
        auto new_row = mesh_coord[0];
        auto new_col = mesh_coord[1];
        if (cluster_axis == 0) {
            new_row = line_index % ring_size;
        } else {
            new_col = line_index % ring_size;
        }
        auto* device = mesh_view.get_device(MeshCoordinate(new_row, new_col));
        TT_FATAL(device != nullptr, "Device not found at coordinate {}", MeshCoordinate(new_row, new_col));
        return device->id();
    };

    bool is_last_chip_in_clockwise_direction = config.device_index == (ring_size - 1);
    bool is_last_chip_in_counter_clockwise_direction = config.device_index == 0;
    config.receiver_device_id =
        is_last_chip_in_clockwise_direction ? std::nullopt : get_chip_id(config.device_index + 1);
    config.sender_device_id =
        is_last_chip_in_counter_clockwise_direction ? std::nullopt : get_chip_id(config.device_index + ring_size - 1);
    return config;
}

std::vector<IDevice*> get_active_physical_devices(const Tensor& tensor) {
    auto* mesh_device = tensor.device();
    std::vector<IDevice*> devices = {};
    devices.reserve(tensor.device_storage().get_coords().size());
    for (const auto& coord : tensor.device_storage().get_coords()) {
        devices.push_back(mesh_device->get_device(coord));
    }
    return devices;
}

std::vector<IDevice*> get_active_physical_devices(const std::vector<Tensor>& tensor_shards) {
    std::vector<IDevice*> devices;
    devices.reserve(tensor_shards.size());
    for (const auto& tensor : tensor_shards) {
        TT_FATAL(
            tensor.device()->shape().mesh_size() == 1,
            "Running a CCL over individual tensor shards requires the shards to be allocated on unit-meshes.");
        devices.push_back(tensor.device()->get_device(MeshCoordinate(0, 0)));
    }
    return devices;
}

std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores(
    size_t num_links,
    size_t num_workers_per_link,
    IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const CoreCoord core_grid_offset,
    const std::optional<CoreRangeSet>& sub_core_grid,
    CoreAllocationStrategy strategy) {
    std::tuple<CoreRangeSet, std::vector<CoreCoord>> result;
    CoreRangeSet sender_worker_core_range;
    const size_t num_workers_preferred = num_workers_per_link * num_links;
    auto available_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
    if (sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(sub_core_grid.value());
    }
    if (available_cores.num_cores() < num_workers_preferred) {
        log_warning(
            tt::LogOp,
            "CCL operation is being launched on a subdevice with fewer worker cores available than ideal. Ideally {} "
            "cores ({} per link and {} links) are made available but only {} are available. This may lead to "
            "performance loss.",
            num_workers_preferred,
            num_workers_per_link,
            num_links,
            available_cores.num_cores());
    }
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;

        if (strategy == CoreAllocationStrategy::COL_MAJOR) {
            // Column-major allocation: fill columns first (outer loop x, inner loop y)
            for (size_t x = start.x; x <= end.x; x++) {
                for (size_t y = start.y; y <= end.y; y++) {
                    sender_worker_core_range = sender_worker_core_range.merge(CoreRangeSet(CoreRange(
                        CoreCoord(x + core_grid_offset.x, y + core_grid_offset.y),
                        CoreCoord(x + core_grid_offset.x, y + core_grid_offset.y))));
                    if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                        break;
                    }
                }
                if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                    break;
                }
            }
        } else {
            // Row-major allocation: fill rows first (outer loop y, inner loop x) - default behavior
            for (size_t y = start.y; y <= end.y; y++) {
                for (size_t x = start.x; x <= end.x; x++) {
                    sender_worker_core_range = sender_worker_core_range.merge(CoreRangeSet(CoreRange(
                        CoreCoord(x + core_grid_offset.x, y + core_grid_offset.y),
                        CoreCoord(x + core_grid_offset.x, y + core_grid_offset.y))));
                    if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                        break;
                    }
                }
                if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                    break;
                }
            }
        }

        if (sender_worker_core_range.num_cores() == num_workers_preferred) {
            break;
        }
    }
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

std::vector<ttnn::Tensor> unpad_output_tensor(
    const std::vector<ttnn::Tensor>& output_tensor,
    const uint32_t num_devices,
    const ttnn::SmallVector<uint32_t>& unpad_elements,
    const int dim) {
    std::vector<ttnn::Tensor> combined_tensors;

    ttnn::SmallVector<uint32_t> begins = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> ends = {1, 1, 1, 1};
    ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    ends = unpad_elements;

    for (int i = 0; i < num_devices; ++i) {
        begins[dim] = i * output_tensor.at(0).logical_shape()[dim] / num_devices;
        ends[dim] = begins[dim] + unpad_elements[dim];

        ttnn::Tensor sliced_tensor = ttnn::slice(output_tensor.at(0), begins, ends, step);

        combined_tensors.push_back(sliced_tensor);
    }
    ttnn::Tensor concat_tensor = ttnn::concat(combined_tensors, dim);
    return {concat_tensor};
}

RingTopology::RingTopology(
    const IDevice* device,
    Topology topology,
    std::optional<uint32_t> sender_device_id,
    std::optional<uint32_t> receiver_device_id,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index) :
    device(device),
    num_links(num_links),
    ring_size(ring_size),
    ring_index(ring_index),
    is_linear(topology == Topology::Linear) {
    eth_sender_cores.reserve(num_links);
    eth_receiver_cores.reserve(num_links);

    uint32_t sender_socket_idx = 0;
    uint32_t receiver_socket_idx = 0;
    if (receiver_device_id == sender_device_id) {
        if (ring_index == 0) {
            receiver_socket_idx = 1;
        } else {
            sender_socket_idx = 1;
        }
    }

    for (uint32_t l = 0; l < num_links; ++l) {
        // Get the cores for the sender and receiver worker cores
        if (!is_linear || ring_index != ring_size - 1) {
            uint32_t receiver_device = receiver_device_id.value();
            const auto& sockets = device->get_ethernet_sockets(receiver_device);
            TT_FATAL(
                sender_socket_idx < sockets.size(),
                "Sender socket index out of bounds. Device has {} ethernet cores but tried to access core at "
                "index {}",
                sockets.size(),
                sender_socket_idx);
            auto eth_sender_core = sockets.at(sender_socket_idx);
            eth_sender_cores.push_back(eth_sender_core);
            log_trace(tt::LogOp, "\teth_sender_core on link {}: (x={},y={})", l, eth_sender_core.x, eth_sender_core.y);
        }
        if (!is_linear || ring_index != 0) {
            uint32_t sender_device = sender_device_id.value();
            const auto& sockets = device->get_ethernet_sockets(sender_device);
            TT_FATAL(
                receiver_socket_idx < sockets.size(),
                "Receiver socket index out of bounds. Device has {} ethernet cores but tried to access core at "
                "index {}",
                sockets.size(),
                receiver_socket_idx);
            auto eth_receiver_core = sockets.at(receiver_socket_idx);
            eth_receiver_cores.push_back(eth_receiver_core);
            log_trace(
                tt::LogOp, "\teth_receiver_core on link {}: (x={},y={})", l, eth_receiver_core.x, eth_receiver_core.y);
        }

        if (receiver_device_id == sender_device_id) {
            receiver_socket_idx += 2;
            sender_socket_idx += 2;
        } else {
            receiver_socket_idx += 1;
            sender_socket_idx += 1;
        }
    }
}

bool RingTopology::is_first_device_in_line(bool in_clockwise_direction) const {
    return this->is_linear && ((in_clockwise_direction && this->ring_index == 0) ||
                               (!in_clockwise_direction && this->ring_index == this->ring_size - 1));
}
bool RingTopology::is_last_device_in_line(bool in_clockwise_direction) const {
    return this->is_linear && ((in_clockwise_direction && this->ring_index == this->ring_size - 1) ||
                               (!in_clockwise_direction && this->ring_index == 0));
}

CclOpTensorConfig::CclOpTensorConfig(const Tensor& tensor) :
    buffer_start_address(tensor.buffer()->address()),
    df(tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype())) {
    if (tensor.layout() == Layout::TILE) {
        this->tile = tensor.tensor_spec().tile();
        this->page_size = this->tile.get_tile_size(this->df);
        this->tile_size = this->tile.get_tile_hw();
    } else {
        this->tile = tt::tt_metal::Tile({32, 32});
        this->page_size = tensor.buffer()->page_size();
        this->tile_size = 1024;
    }
}
uint32_t CclOpTensorConfig::get_page_size() const { return this->page_size; }
uint32_t CclOpTensorConfig::get_tile_size() const { return this->tile_size; }
tt::tt_metal::Tile CclOpTensorConfig::get_tile() const { return this->tile; }

uint32_t CclOpTensorConfig::get_buffer_start_address() const { return this->buffer_start_address; }

CclOpInterleavedTensorConfig::CclOpInterleavedTensorConfig(const Tensor& input_tensor) :
    CclOpTensorConfig(input_tensor) {}

CclOpShardedTensorConfig::CclOpShardedTensorConfig(const Tensor& tensor) :
    CclOpTensorConfig(tensor), shard_spec(tensor.shard_spec().value()) {}

const tt::tt_metal::ShardSpec& CclOpShardedTensorConfig::get_shard_spec() const { return this->shard_spec; }

std::unique_ptr<CclOpTensorConfig> CclOpTensorConfig::build_all_gather_tensor_config(const Tensor& tensor) {
    if (tensor.is_sharded()) {
        return std::make_unique<CclOpShardedTensorConfig>(tensor);
    }
    return std::make_unique<CclOpInterleavedTensorConfig>(tensor);
}

void generate_edm_kernels_for_ring_or_linear_topology(
    Program& program,
    const IDevice* device,
    const RingTopology& topology_config,
    const std::vector<ccl::EriscDatamoverBuilder>& clockwise_edm_builders,
    const std::vector<ccl::EriscDatamoverBuilder>& counter_clockwise_edm_builders,
    std::optional<uint32_t> /*receiver_device_id*/,
    std::optional<uint32_t> /*sender_device_id*/) {
    auto sender_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(tt::tt_metal::hal::get_arch());
    auto receiver_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(tt::tt_metal::hal::get_arch());
    for (uint32_t i = 0; i < topology_config.num_links; ++i) {
        bool is_clockwise_direction_edm_enabled =
            !topology_config.is_linear || topology_config.ring_index != topology_config.ring_size - 1;
        if (is_clockwise_direction_edm_enabled) {
            auto eth_sender_core = topology_config.eth_sender_cores.at(i);
            log_trace(tt::LogOp, "EDM CLOCKWISE KERNEL RT ARGS: ");
            generate_edm_kernel(
                program,
                device,
                clockwise_edm_builders.at(i),
                eth_sender_core,
                tt::tt_metal::DataMovementProcessor::RISCV_0,
                sender_noc);
            log_trace(
                tt::LogOp,
                "RingIndex: {}. Link {}. Clockwise EDM Core (x={},y={})",
                topology_config.ring_index,
                i,
                eth_sender_core.x,
                eth_sender_core.y);
        }

        bool is_counter_clockwise_direction_edm_enabled = !topology_config.is_linear || topology_config.ring_index != 0;
        if (is_counter_clockwise_direction_edm_enabled) {
            log_trace(tt::LogOp, "EDM COUNTER CLOCKWISE KERNEL RT ARGS: ");
            auto eth_receiver_core = topology_config.eth_receiver_cores.at(i);
            generate_edm_kernel(
                program,
                device,
                counter_clockwise_edm_builders.at(i),
                eth_receiver_core,
                tt::tt_metal::DataMovementProcessor::RISCV_0,
                receiver_noc);
            log_trace(
                tt::LogOp,
                "RingIndex: {}. Link {}. Counter-clockwise EDM Core (x={},y={})",
                topology_config.ring_index,
                i,
                eth_receiver_core.x,
                eth_receiver_core.y);
        }
    }
}

tt::tt_metal::KernelHandle generate_edm_kernel(
    Program& program,
    const IDevice* /*device*/,
    const ccl::EriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core,
    const tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id) {
    edm_builder.dump_to_log();
    return tt::tt_fabric::generate_erisc_datamover_kernel(tt::tt_fabric::FabricEriscDatamoverKernelConfig{
        .program = program,
        .kernel_path = "ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp",
        .eth_core = eth_core,
        .risc_id = risc_id,
        .noc_id = noc_id,
        .compile_time_args = edm_builder.get_compile_time_args((uint32_t)risc_id),
        .named_compile_time_args = {},
        .runtime_args = edm_builder.get_runtime_args(),
        .opt_level = std::nullopt,
    });
}

ccl::EriscDatamoverBuilder create_erisc_datamover_builder(
    std::size_t num_channels,
    uint32_t page_size,
    size_t num_buffers_per_channel,
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode,
    ccl::EriscDataMoverTerminationMode termination_mode) {
    ccl::EriscDatamoverConfig config;
    TT_ASSERT(num_channels > 0);
    std::vector<uint32_t> edm_sem_addresses(num_channels, 0);
    std::vector<uint32_t> edm_buffer_addresses(num_channels, 0);

    uint32_t edm_sem_addr = config.get_semaphores_base_address(num_channels);
    uint32_t edm_buffer_addr = config.get_buffers_base_address(num_channels);
    TT_ASSERT(edm_sem_addr > 0);
    TT_ASSERT(edm_buffer_addr > 0);
    const uint32_t channel_buffer_size = config.compute_buffer_size(num_channels, num_buffers_per_channel, page_size);
    for (std::size_t c = 0; c < num_channels; ++c) {
        edm_sem_addresses.at(c) = edm_sem_addr;
        edm_sem_addr += ccl::EriscDatamoverConfig::semaphore_size;
        TT_ASSERT(edm_buffer_addr % EriscDatamoverConfig::get_eth_word_size() == 0);
        edm_buffer_addresses.at(c) = edm_buffer_addr;
        log_trace(tt::LogOp, " edm_buffer_addresses({}) = {}", c, edm_buffer_addr);
        edm_buffer_addr += num_buffers_per_channel *
                           (channel_buffer_size + (ccl::EriscDatamoverConfig::enable_merged_payload_and_channel_sync
                                                       ? ccl::EriscDatamoverConfig::get_eth_channel_sync_size_bytes()
                                                       : 0));
        TT_ASSERT((c == 0) || (edm_buffer_addresses.back() != edm_buffer_addresses.front()));
        TT_ASSERT((c == 0) || (edm_sem_addresses.back() != edm_sem_addresses.front()));
    }

    return ccl::EriscDatamoverBuilder(
        channel_buffer_size,
        config.get_edm_handshake_address(),
        edm_sem_addresses,
        edm_buffer_addresses,
        buffer_sharing_mode,
        termination_mode,
        num_buffers_per_channel);
}

template <class DERIVED_SLICER_T>
RingReduceScatterBaseTensorSlicer<DERIVED_SLICER_T>::RingReduceScatterBaseTensorSlicer(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    int slice_dim,
    uint32_t /*ring_index*/,
    uint32_t ring_size,
    uint32_t total_num_workers,
    uint32_t max_slice_size_in_bytes,
    uint32_t half_cb_n_pages) :
    LegacyCclTensorSlicer() {
    TT_ASSERT(max_slice_size_in_bytes > 0);
    TT_ASSERT(input_tensor.padded_shape().rank() == 4);
    this->row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    this->slice_dim_is_width = input_tensor.padded_shape().rank() - 1 == slice_dim;
    this->is_sharded = input_tensor.is_sharded();

    this->input_page_size = input_tensor.buffer()->page_size();
    log_trace(tt::LogOp, "input_page_size={}", input_page_size);
    if (row_major) {
        this->num_cols = input_tensor.padded_shape()[-1];
        const auto& input_shape = input_tensor.padded_shape();
        const auto& output_shape = output_tensor.padded_shape();
        this->num_rows =
            std::accumulate(input_shape.cbegin() + slice_dim, input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
        this->row_offset =
            std::accumulate(
                output_shape.cbegin() + slice_dim, output_shape.cend() - 1, 1, std::multiplies<uint32_t>()) -
            num_rows;
    } else {
        auto input_tile = input_tensor.tensor_spec().tile();
        const uint32_t num_tiles_x = input_tensor.padded_shape()[-1] / input_tile.get_width();
        uint32_t num_tiles_y = (input_tensor.padded_shape()[-2] / input_tile.get_height());
        for (std::size_t i = 0; input_tensor.padded_shape().rank() > 2 && i < input_tensor.padded_shape().rank() - 2;
             i++) {
            num_tiles_y *= input_tensor.padded_shape()[i];
        }
        TT_ASSERT(num_tiles_x >= ring_size);
        this->tensor_slice_shape.x = slice_dim == 3 ? (num_tiles_x / ring_size) : num_tiles_x;
        this->tensor_slice_shape.y = slice_dim != 3 ? num_tiles_y / ring_size : num_tiles_y;
    }

    // Create the worker schedule

    // The `output_page_offset` will be the starting page offset for this slice index (corresponds to )
    // ring index). Each worker will operate out of that slice and then advance to the next slice for
    // for the next ring index/timestep
    if (row_major) {
        if (slice_dim_is_width) {
            TT_THROW("Reduce scatter row-major interleaved does not yet support a width dim");
            this->output_addr_offset = input_page_size;
        } else {
            this->output_page_offset = num_rows;
        }
        this->worker_slice_shapes = create_worker_slice_shapes_for_row_major_layout(
            this->tensor_slice_shape, total_num_workers, max_slice_size_in_bytes);
    } else {
        log_trace(tt::LogOp, "\tmax_slice_size_in_bytes={}", max_slice_size_in_bytes);
        log_trace(tt::LogOp, "\tinput_page_size={}", input_page_size);
        this->worker_slice_shapes = DERIVED_SLICER_T::create_worker_slice_shapes_for_tile_layout(
            input_tensor.padded_shape(),
            this->tensor_slice_shape,
            total_num_workers,
            max_slice_size_in_bytes / input_page_size,
            half_cb_n_pages);
    }

    if (row_major) {
        this->flattened_tensor_shape = tt_xy_pair{
            input_tensor.padded_shape()[3],
            input_tensor.padded_shape()[0] * input_tensor.padded_shape()[1] * input_tensor.padded_shape()[2]};
    } else {
        auto input_tile = input_tensor.tensor_spec().tile();
        this->flattened_tensor_shape = tt_xy_pair{
            input_tensor.padded_shape()[3] / input_tile.get_width(),
            (input_tensor.padded_shape()[0] * input_tensor.padded_shape()[1] * input_tensor.padded_shape()[2]) /
                input_tile.get_height()};
    }

    this->worker_slice_offsets =
        DERIVED_SLICER_T::compute_worker_slice_offsets(this->worker_slice_shapes, this->tensor_slice_shape);
    TT_ASSERT(this->worker_slice_offsets.size() == this->worker_slice_shapes.size());
}

RingReduceScatterTensorSlicer::RingReduceScatterTensorSlicer(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    int slice_dim,
    uint32_t ring_index,
    uint32_t ring_size,
    uint32_t total_num_workers,
    uint32_t max_slice_size_in_bytes,
    uint32_t half_cb_n_pages) :
    RingReduceScatterBaseTensorSlicer<RingReduceScatterTensorSlicer>(
        input_tensor,
        output_tensor,
        slice_dim,
        ring_index,
        ring_size,
        total_num_workers,
        max_slice_size_in_bytes,
        half_cb_n_pages) {};

RingReduceScatterWrappedTensorSlicer::RingReduceScatterWrappedTensorSlicer(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    int slice_dim,
    uint32_t ring_index,
    uint32_t ring_size,
    uint32_t total_num_workers,
    uint32_t max_slice_size_in_bytes,
    uint32_t half_cb_n_pages) :
    RingReduceScatterBaseTensorSlicer<RingReduceScatterWrappedTensorSlicer>(
        input_tensor,
        output_tensor,
        slice_dim,
        ring_index,
        ring_size,
        total_num_workers,
        max_slice_size_in_bytes,
        half_cb_n_pages) {};

std::vector<tt_xy_pair> RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
    const std::vector<tt_xy_pair>& worker_slice_shapes, const tt_xy_pair& tensor_slice_shape) {
    std::vector<tt_xy_pair> worker_slice_offsets;
    worker_slice_offsets.reserve(worker_slice_shapes.size());

    std::size_t offset_x = 0;
    std::size_t offset_y = 0;
    std::size_t last_worker_size_y = worker_slice_shapes.at(0).y;  // for validation
    bool first_in_row = true;
    for (const tt_xy_pair& worker_slice_shape : worker_slice_shapes) {
        worker_slice_offsets.emplace_back(offset_x, offset_y);

        TT_ASSERT(offset_y < tensor_slice_shape.y);
        offset_x += worker_slice_shape.x;
        if (offset_x < tensor_slice_shape.x) {
            first_in_row = false;
        } else {
            offset_x = 0;
            first_in_row = true;
            offset_y += worker_slice_shape.y;
        }
        TT_ASSERT(first_in_row || last_worker_size_y == worker_slice_shape.y);
        last_worker_size_y = worker_slice_shape.y;
    }

    TT_ASSERT(worker_slice_offsets.size() == worker_slice_shapes.size());
    return worker_slice_offsets;
}

static std::vector<tt_xy_pair> compute_worker_slice_offsets_for_wrapped_tensor_slicer(
    const std::vector<tt_xy_pair>& worker_slice_shapes, const tt_xy_pair& tensor_slice_shape) {
    std::vector<tt_xy_pair> worker_slice_offsets;
    worker_slice_offsets.reserve(worker_slice_shapes.size());

    std::uint32_t flattened_idx = 0;

    for (const tt_xy_pair& worker_slice_shape : worker_slice_shapes) {
        // Convert from flat to (x, y) coordinates
        std::size_t offset_x = flattened_idx % tensor_slice_shape.x;
        std::size_t offset_y = flattened_idx / tensor_slice_shape.x;

        // Append the offset to the list
        worker_slice_offsets.emplace_back(offset_x, offset_y);

        // Update the flattened index
        flattened_idx += worker_slice_shape.x * worker_slice_shape.y;
    }

    TT_ASSERT(worker_slice_offsets.size() == worker_slice_shapes.size());
    return worker_slice_offsets;
}

std::vector<tt_xy_pair> RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(
    const std::vector<tt_xy_pair>& worker_slice_shapes, const tt_xy_pair& tensor_slice_shape) {
    return compute_worker_slice_offsets_for_wrapped_tensor_slicer(worker_slice_shapes, tensor_slice_shape);
}

template <class DERIVED_SLICER_T>
std::vector<tt_xy_pair>
RingReduceScatterBaseTensorSlicer<DERIVED_SLICER_T>::create_worker_slice_shapes_for_row_major_layout(
    const tt_xy_pair& tensor_slice_shape_in_elems, uint32_t num_workers, uint32_t max_slice_size_in_elements) {
    std::vector<tt_xy_pair> worker_slice_shapes;
    worker_slice_shapes.reserve(num_workers);

    if (num_workers > tensor_slice_shape_in_elems.y) {
        log_warning(
            tt::LogOp,
            "Reduce Scatter more workers instantiated than is work to be done. Some workers will be idle and do "
            "nothing");
        num_workers = tensor_slice_shape_in_elems.y;
        for (uint32_t w = 0; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(tensor_slice_shape_in_elems.x, 1);
        }
        for (uint32_t w = num_workers; w < tensor_slice_shape_in_elems.x; ++w) {
            worker_slice_shapes.emplace_back(0, 0);
        }
        return worker_slice_shapes;
    }

    uint32_t num_elems_accounted_for = 0;
    // For now we don't support row splitting but we will in the future
    const uint32_t min_rows_per_worker = tensor_slice_shape_in_elems.y / num_workers;
    const uint32_t num_workers_with_max_rows = tensor_slice_shape_in_elems.y % num_workers;
    const uint32_t max_rows_per_worker = num_workers_with_max_rows != 0 ? min_rows_per_worker + 1 : min_rows_per_worker;
    for (uint32_t w = 0; w < num_workers_with_max_rows; w++) {
        worker_slice_shapes.emplace_back(tensor_slice_shape_in_elems.x, max_rows_per_worker);
        num_elems_accounted_for += tensor_slice_shape_in_elems.x * max_rows_per_worker;
    }
    for (uint32_t w = num_workers_with_max_rows; w < num_workers; w++) {
        worker_slice_shapes.emplace_back(tensor_slice_shape_in_elems.x, min_rows_per_worker);
        num_elems_accounted_for += tensor_slice_shape_in_elems.x * min_rows_per_worker;
    }

    TT_ASSERT(num_elems_accounted_for == tensor_slice_shape_in_elems.x * tensor_slice_shape_in_elems.y);
    for (auto& worker_slice_shape : worker_slice_shapes) {
        TT_ASSERT(max_slice_size_in_elements >= worker_slice_shape.x * worker_slice_shape.y);
        TT_ASSERT(worker_slice_shape.x * worker_slice_shape.y > 0);
    }
    return worker_slice_shapes;
}

std::vector<tt_xy_pair> RingReduceScatterTensorSlicer::create_worker_slice_shapes_for_tile_layout(
    const ttnn::Shape& tensor_shape,
    const tt_xy_pair& tensor_slice_shape_in_tiles,
    uint32_t num_workers,
    uint32_t max_slice_size_in_pages,
    uint32_t half_cb_n_pages) {
    log_trace(tt::LogOp, "\tmax_slice_size_in_pages={}", max_slice_size_in_pages);
    TT_ASSERT(max_slice_size_in_pages > 0);
    std::vector<tt_xy_pair> worker_slice_shapes;
    worker_slice_shapes.reserve(num_workers);
    const uint32_t total_num_tiles = tensor_slice_shape_in_tiles.x * tensor_slice_shape_in_tiles.y;
    if (num_workers > total_num_tiles) {
        log_warning(
            tt::LogOp,
            "Reduce Scatter more workers instantiated than is work to be done. Some workers will be idle and do "
            "nothing");
        for (uint32_t w = 0; w < total_num_tiles; ++w) {
            worker_slice_shapes.emplace_back(1, 1);
        }
        for (uint32_t w = total_num_tiles; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(0, 0);
        }
        return worker_slice_shapes;
    }

    std::size_t max_slice_size_in_tiles = max_slice_size_in_pages;
    // Add padding for filler pages

    TT_ASSERT(max_slice_size_in_tiles > 0);

    uint32_t num_tiles_accounted_for = 0;  // for validation
    if (tensor_slice_shape_in_tiles.y >= num_workers) {
        // slice into rows
        const uint32_t min_rows_per_worker = tensor_slice_shape_in_tiles.y / num_workers;
        const uint32_t num_workers_with_max_rows = tensor_slice_shape_in_tiles.y % num_workers;
        const uint32_t max_rows_per_worker =
            num_workers_with_max_rows != 0 ? min_rows_per_worker + 1 : min_rows_per_worker;
        for (uint32_t w = 0; w < num_workers_with_max_rows; w++) {
            worker_slice_shapes.emplace_back(tensor_slice_shape_in_tiles.x, max_rows_per_worker);
            num_tiles_accounted_for += tensor_slice_shape_in_tiles.x * max_rows_per_worker;
        }
        for (uint32_t w = num_workers_with_max_rows; w < num_workers; w++) {
            worker_slice_shapes.emplace_back(tensor_slice_shape_in_tiles.x, min_rows_per_worker);
            num_tiles_accounted_for += tensor_slice_shape_in_tiles.x * min_rows_per_worker;
        }
    } else if (tensor_slice_shape_in_tiles.x >= num_workers) {
        // slice into columns
        const uint32_t min_cols_per_worker = tensor_slice_shape_in_tiles.x / num_workers;
        const uint32_t num_workers_with_max_cols = tensor_slice_shape_in_tiles.x % num_workers;
        const uint32_t max_cols_per_worker =
            num_workers_with_max_cols != 0 ? min_cols_per_worker + 1 : min_cols_per_worker;
        for (uint32_t w = 0; w < num_workers_with_max_cols; w++) {
            worker_slice_shapes.emplace_back(max_cols_per_worker, tensor_slice_shape_in_tiles.y);
            num_tiles_accounted_for += max_cols_per_worker * tensor_slice_shape_in_tiles.y;
        }
        for (uint32_t w = num_workers_with_max_cols; w < num_workers; w++) {
            worker_slice_shapes.emplace_back(min_cols_per_worker, tensor_slice_shape_in_tiles.y);
            num_tiles_accounted_for += min_cols_per_worker * tensor_slice_shape_in_tiles.y;
        }

    } else {
        const uint32_t min_num_workers_per_row = num_workers / tensor_slice_shape_in_tiles.y;
        const uint32_t num_rows_with_max_workers = tensor_slice_shape_in_tiles.y % num_workers;
        const uint32_t max_num_workers_per_row =
            num_rows_with_max_workers != 0 ? min_num_workers_per_row + 1 : min_num_workers_per_row;

        // 4 "quadrants" to the worker slicing:
        // 1. Row with max num workers and max columns wide per worker (first part of rows with max num workers)
        // 2. Row with max num workers and min columns wide per worker (second part of rows with max num workers)
        // 3. Row with min num workers and max columns wide per worker (first part of rows with min num workers)
        // 4. Row with min num workers and min columns wide per worker (second part of rows with min num workers)
        // Depending on specific numbers, some of the above "quadrants" might be 0 sized
        const uint32_t max_workers_row_min_cols_per_worker = tensor_slice_shape_in_tiles.x / max_num_workers_per_row;
        const uint32_t max_workers_row_max_col_worker_count = tensor_slice_shape_in_tiles.x % max_num_workers_per_row;
        const uint32_t max_workers_row_max_cols_per_worker = max_workers_row_max_col_worker_count != 0
                                                                 ? max_workers_row_min_cols_per_worker + 1
                                                                 : max_workers_row_min_cols_per_worker;
        TT_ASSERT(max_workers_row_min_cols_per_worker > 0);
        TT_ASSERT(max_workers_row_max_cols_per_worker >= max_workers_row_min_cols_per_worker);
        for (uint32_t w_r = 0; w_r < num_rows_with_max_workers; w_r++) {
            for (uint32_t w_c = 0; w_c < max_workers_row_max_cols_per_worker; w_c++) {
                worker_slice_shapes.emplace_back(max_workers_row_max_cols_per_worker, 1);
                num_tiles_accounted_for += max_workers_row_max_cols_per_worker;
            }
            for (uint32_t w_c = max_workers_row_max_col_worker_count; w_c < max_num_workers_per_row; w_c++) {
                worker_slice_shapes.emplace_back(max_workers_row_min_cols_per_worker, 1);
                num_tiles_accounted_for += max_workers_row_min_cols_per_worker;
            }
        }

        TT_ASSERT(min_num_workers_per_row > 0);
        const uint32_t min_workers_row_min_cols_per_worker = tensor_slice_shape_in_tiles.x / min_num_workers_per_row;
        const uint32_t min_workers_row_max_col_worker_count = tensor_slice_shape_in_tiles.x % min_num_workers_per_row;
        const uint32_t min_workers_row_max_cols_per_worker = min_workers_row_max_col_worker_count != 0
                                                                 ? min_workers_row_min_cols_per_worker + 1
                                                                 : min_workers_row_min_cols_per_worker;

        for (uint32_t w_r = num_rows_with_max_workers; w_r < tensor_slice_shape_in_tiles.y; w_r++) {
            for (uint32_t w_c = 0; w_c < min_workers_row_max_cols_per_worker; w_c++) {
                worker_slice_shapes.emplace_back(min_workers_row_max_cols_per_worker, 1);
                num_tiles_accounted_for += min_workers_row_max_cols_per_worker;
            }
            for (uint32_t w_c = min_workers_row_max_col_worker_count; w_c < min_num_workers_per_row; w_c++) {
                worker_slice_shapes.emplace_back(min_workers_row_min_cols_per_worker, 1);
                num_tiles_accounted_for += min_workers_row_max_cols_per_worker;
            }
        }
    }

    // For now we do something a little naive - since this becomes an optimization problem otherwise, and the
    // benefits to nailing it are marginal we expect uniform chunk sizes and just truncate the largest chunk to fit
    // the max size and then apply that shape to all workers slice shapes
    tt_xy_pair largest_worker_slice_shape = {0, 0};
    for (const auto& worker_slice_shape : worker_slice_shapes) {
        if (largest_worker_slice_shape.x * largest_worker_slice_shape.y < worker_slice_shape.x * worker_slice_shape.y) {
            largest_worker_slice_shape = worker_slice_shape;
        }
    }

    // This is a bit of a hack for now until we support true 4D shapes in our slicer and our indexer (device side)
    bool has_gt_1_depth_size = false;
    for (std::size_t i = 0; tensor_shape.rank() > 2 && i < tensor_shape.rank() - 2; i++) {
        has_gt_1_depth_size = has_gt_1_depth_size || tensor_shape[i] > 1;
    }
    if (has_gt_1_depth_size) {
        largest_worker_slice_shape.y = 1;
    }

    bool do_truncation = ((largest_worker_slice_shape.x * largest_worker_slice_shape.y) > max_slice_size_in_tiles) ||
                         has_gt_1_depth_size;
    if (do_truncation) {
        log_trace(tt::LogOp, "Truncating worker slice shapes to fit max slice size in tiles");
    }
    log_trace(
        tt::LogOp,
        "largest_worker_slice_shape: x={}, y={}",
        largest_worker_slice_shape.x,
        largest_worker_slice_shape.y);
    log_trace(tt::LogOp, "max_slice_size_in_tiles={}", max_slice_size_in_tiles);
    auto get_padded_worker_slice_size_in_tiles = [](const tt_xy_pair& worker_slice_shape, uint32_t half_cb_n_pages) {
        return tt::round_up(worker_slice_shape.x * worker_slice_shape.y, half_cb_n_pages);
    };

    while (get_padded_worker_slice_size_in_tiles(largest_worker_slice_shape, half_cb_n_pages) >
           max_slice_size_in_tiles) {
        log_trace(tt::LogOp, "Loop Head");
        // truncate the largest dim first
        uint32_t delta = (largest_worker_slice_shape.x * largest_worker_slice_shape.y) - max_slice_size_in_tiles;
        log_trace(tt::LogOp, "-- delta: {}", delta);
        uint32_t cols_removed_if_x_truncated = std::max<uint32_t>(1, largest_worker_slice_shape.x / delta);
        uint32_t tiles_removed_if_x_truncated = cols_removed_if_x_truncated * largest_worker_slice_shape.y;
        uint32_t rows_removed_if_y_truncated = std::max<uint32_t>(1, largest_worker_slice_shape.y / delta);
        uint32_t tiles_removed_if_y_truncated = rows_removed_if_y_truncated * largest_worker_slice_shape.x;
        uint32_t difference_x = tiles_removed_if_x_truncated > delta ? tiles_removed_if_x_truncated - delta
                                                                     : delta - tiles_removed_if_x_truncated;
        uint32_t difference_y = tiles_removed_if_y_truncated > delta ? tiles_removed_if_y_truncated - delta
                                                                     : delta - tiles_removed_if_y_truncated;
        log_trace(tt::LogOp, "-- cols_removed_if_x_truncated: {}", cols_removed_if_x_truncated);
        log_trace(tt::LogOp, "-- tiles_removed_if_x_truncated: {}", tiles_removed_if_x_truncated);
        log_trace(tt::LogOp, "-- rows_removed_if_y_truncated: {}", rows_removed_if_y_truncated);
        log_trace(tt::LogOp, "-- tiles_removed_if_y_truncated: {}", tiles_removed_if_y_truncated);
        log_trace(tt::LogOp, "-- difference_x: {}", difference_x);
        log_trace(tt::LogOp, "-- difference_y: {}", difference_y);
        if (difference_x < difference_y) {
            largest_worker_slice_shape.x -= cols_removed_if_x_truncated;
        } else {
            largest_worker_slice_shape.y -= rows_removed_if_y_truncated;
        }
        log_trace(
            tt::LogOp,
            "-- new largest_worker_slice_shape: x={}, y={}",
            largest_worker_slice_shape.x,
            largest_worker_slice_shape.y);
    }
    if (do_truncation) {
        log_trace(
            tt::LogOp,
            "Truncated worker slice shape to fit max slice size in tiles: ({},{})",
            largest_worker_slice_shape.x,
            largest_worker_slice_shape.y);
        if (!(largest_worker_slice_shape.x * largest_worker_slice_shape.y > 0)) {
            log_warning(
                tt::LogOp,
                "Computing worker slice shape for reduce scatter resulted in 0 sized slice. Defaulting to 1x1 page per "
                "worker, which is likely to lead to suboptimal performance");
            largest_worker_slice_shape.x = 1;
            largest_worker_slice_shape.y = 1;
        }
        TT_ASSERT(largest_worker_slice_shape.x * largest_worker_slice_shape.y > 0);
        for (auto& worker_slice_shape : worker_slice_shapes) {
            worker_slice_shape = largest_worker_slice_shape;
        }
    }

    TT_ASSERT(num_tiles_accounted_for == total_num_tiles, "All tiles must be accounted for in the worker slice shapes");
    TT_ASSERT(worker_slice_shapes.size() == num_workers, "Worker slice shapes must match the number of workers");
    std::for_each(
        worker_slice_shapes.begin(),
        worker_slice_shapes.end(),
        [max_slice_size_in_pages](const tt_xy_pair& worker_slice_shape) {
            TT_ASSERT(worker_slice_shape.x * worker_slice_shape.y <= max_slice_size_in_pages);
        });
    return worker_slice_shapes;
}

std::vector<tt_xy_pair> RingReduceScatterWrappedTensorSlicer::create_worker_slice_shapes_for_tile_layout(
    const ttnn::Shape& /*tensor_shape*/,
    const tt_xy_pair& tensor_slice_shape_in_tiles,
    uint32_t num_workers,
    uint32_t max_slice_size_in_pages,
    uint32_t /*half_cb_n_pages*/) {
    log_trace(tt::LogOp, "\tmax_slice_size_in_pages={}", max_slice_size_in_pages);
    TT_ASSERT(max_slice_size_in_pages > 0);
    std::vector<tt_xy_pair> worker_slice_shapes;
    worker_slice_shapes.reserve(num_workers);
    const uint32_t total_num_tiles = tensor_slice_shape_in_tiles.x * tensor_slice_shape_in_tiles.y;
    if (num_workers > total_num_tiles) {
        log_warning(
            tt::LogOp,
            "Reduce Scatter more workers instantiated than is work to be done. Some workers will be idle and do "
            "nothing");
        for (uint32_t w = 0; w < total_num_tiles; ++w) {
            worker_slice_shapes.emplace_back(1, 1);
        }
        for (uint32_t w = total_num_tiles; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(0, 0);
        }
        return worker_slice_shapes;
    }

    std::size_t max_slice_size_in_tiles = max_slice_size_in_pages;

    // Assign slices by assuming that the input tensor is flattened into a 1D Shape
    // Cast to double before division to ensure ceil() rounds up properly.
    // Example: 10 tiles / 3 workers: was ceil(3)=3, now ceil(3.33)=4 tiles per worker.
    std::size_t optim_worker_slice_len_tiles = static_cast<std::size_t>(ceil(
        static_cast<double>(total_num_tiles) /
        num_workers));  // Ceil so that the remainder worker will have a smaller slice

    if (max_slice_size_in_tiles < optim_worker_slice_len_tiles) {  // Each worker will have a full slice
        for (uint32_t w = 0; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(max_slice_size_in_tiles, 1);
        }
    } else {  // Each worker will only have one slice
        uint32_t remainder_worker_len_tiles = total_num_tiles % optim_worker_slice_len_tiles;

        for (uint32_t w = 0; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(optim_worker_slice_len_tiles, 1);
        }
        // If there is a remainder worker, we need to adjust the last worker's slice shape to be smaller
        if (remainder_worker_len_tiles > 0) {
            worker_slice_shapes.back() = tt_xy_pair{remainder_worker_len_tiles, 1};
        }
    }

    return worker_slice_shapes;
}

/*
 * @brief: Given a tensor shape, evenly break it into pieces along a given dimension and generate the slices
 * accordingly. This can be fed into a CCL Send command generator
 */
std::vector<TensorSlice> generate_slice_sequence_on_dim(
    TensorSlice::ords_t tensor_shape,
    TensorSlice::ords_t worker_slice_shape,
    std::size_t fracture_dim,
    std::size_t num_slices,
    std::int64_t start_slice_index,
    std::int64_t end_slice_index_exclusive,
    std::size_t worker_index) {
    static_assert(
        std::is_same_v<TensorSlice::ords_t, tt_xy_pair>,
        "generate_slice_sequence_on_dim not yet implemented for type not of tt_xy_pair");
    // We don't support 4D shapes in the CCL kernels yet, which are needed for proper reduction/concatenation in some
    // cases so for now we subtract the outer dims from the fracture_dim since we only support 2D at the moment.
    if (fracture_dim == 3) {
        fracture_dim -= 2;
    } else {
        // dims are
        fracture_dim = 0;
    }

    TT_ASSERT(worker_slice_shape.y == 1);

    std::vector<TensorSlice> slices;
    auto dim_size = fracture_dim == 1 ? tensor_shape.x : tensor_shape.y;
    TT_ASSERT(dim_size % num_slices == 0);
    auto slice_size_on_dim = dim_size / num_slices;
    auto slice_shape = fracture_dim == 0 ? tt_xy_pair{tensor_shape.x, slice_size_on_dim}
                                         : tt_xy_pair{slice_size_on_dim, tensor_shape.y};

    auto dim_start_offset = start_slice_index * slice_size_on_dim;
    TensorSlice::ords_t tensor_slice_offset =
        fracture_dim == 0 ? tt_xy_pair{0, dim_start_offset} : tt_xy_pair{dim_start_offset, 0};

    bool forward_direction = start_slice_index > end_slice_index_exclusive;  // only for debug
    auto incr = start_slice_index < end_slice_index_exclusive ? 1 : -1;
    if (forward_direction) {
        log_trace(tt::LogOp, "slice_size_on_dim {}", slice_size_on_dim);
        log_trace(tt::LogOp, "worker_index {}", worker_index);
    }

    auto worker_slice_start_offset =
        /*fracture_dim == 0 ? TensorSlice::ords_t{0, worker_index * worker_slice_shape.y} :*/ TensorSlice::ords_t{
            worker_index * worker_slice_shape.x, 0};

    auto generate_slice = [forward_direction,
                           incr,
                           &slices,
                           &tensor_shape,
                           &slice_shape,
                           &worker_slice_shape,
                           tensor_slice_offset,
                           &worker_slice_start_offset,
                           fracture_dim,
                           dim_start_offset,
                           slice_size_on_dim](std::int64_t i) {
        auto tensor_slice_offset_adjusted = tensor_slice_offset;
        if (fracture_dim == 0) {
            tensor_slice_offset_adjusted.y = slice_size_on_dim * i;
        } else {
            tensor_slice_offset_adjusted.x = slice_size_on_dim * i;
        }
        TT_ASSERT(tensor_shape.x > 0, "Invalid tensor shape. x = 0 but it must be > 0");
        TT_ASSERT(tensor_shape.y > 0, "Invalid tensor shape. y = 0 but it must be > 0");
        TT_ASSERT(slice_shape.x > 0, "Invalid tensor slice shape. x = 0 but it must be > 0");
        TT_ASSERT(slice_shape.y > 0, "Invalid tensor slice shape. x = 0 but it must be > 0");
        TT_ASSERT(
            tensor_slice_offset_adjusted.x < tensor_shape.x,
            "Invalid tensor slice offset. x = {} but it must be < tensor shape x={}. slice_offset: (y={},x={}), "
            "tensor_shape: (y={},x={}). slice_size_on_dim: {}, i: {}",
            tensor_slice_offset_adjusted.x,
            tensor_shape.x,
            tensor_slice_offset_adjusted.y,
            tensor_slice_offset_adjusted.x,
            tensor_shape.y,
            tensor_shape.x,
            slice_size_on_dim,
            i);
        TT_ASSERT(
            tensor_slice_offset_adjusted.y < tensor_shape.y,
            "Invalid tensor slice offset. y = {} but it must be < tensor shape y={}. slice_offset: (y={},x={}), "
            "tensor_shape: (y={},x={}). slice_size_on_dim: {}, i: {}",
            tensor_slice_offset_adjusted.y,
            tensor_shape.y,
            tensor_slice_offset_adjusted.y,
            tensor_slice_offset_adjusted.x,
            tensor_shape.y,
            tensor_shape.x,
            slice_size_on_dim,
            i);
        TT_ASSERT(worker_slice_shape.x > 0, "Invalid worker slice shape. x = 0 but it must be > 0");
        TT_ASSERT(worker_slice_shape.y > 0, "Invalid worker slice shape. y = 0 but it must be > 0");

        const auto& tensor_slice = TensorSlice(
            tensor_shape,
            slice_shape,
            tensor_slice_offset_adjusted,
            worker_slice_shape,
            worker_slice_start_offset,
            fracture_dim);
        if (forward_direction) {
            log_trace(
                tt::LogOp,
                "generate_slice ({}):\n\ttensor_shape: (y={},x={})\n\ttensor_slice_shape: "
                "(y={},x={})\n\ttensor_slice_offset_adjusted: (y={},x={})\n\tslice_start_shape: (y={},x={})\n\tworker "
                "relative slice_start_offset: (y={},x={})\n\tfracture_dim: {}\n\tdim_start_offset: "
                "{}\n\tslice_size_on_dim: {}\n",
                i,
                tensor_slice.tensor_shape.y,
                tensor_slice.tensor_shape.x,
                tensor_slice.tensor_slice_shape.y,
                tensor_slice.tensor_slice_shape.x,
                tensor_slice.tensor_slice_offset.y,
                tensor_slice.tensor_slice_offset.x,
                tensor_slice.worker_slice_shape.y,
                tensor_slice.worker_slice_shape.x,
                tensor_slice.worker_slice_offset.y,
                tensor_slice.worker_slice_offset.x,
                fracture_dim,
                dim_start_offset,
                slice_size_on_dim);
        }

        slices.push_back(tensor_slice);
    };

    for (int i = start_slice_index; i != end_slice_index_exclusive; i += incr) {
        generate_slice(i);
    }

    return slices;
}

/*
 * @brief: Given a tensor shape, evenly break it into pieces along a given dimension and generate the slices
 * accordingly. This can be fed into a CCL Send command generator
 */
std::vector<TensorSlice> generate_slice_sequence_on_dim_v2(
    TensorSlice::ords_t tensor_shape,
    TensorSlice::ords_t worker_slice_shape,
    TensorSlice::ords_t worker_slice_offset,
    std::size_t fracture_dim,
    std::size_t num_slices,
    std::int64_t start_slice_index,
    std::int64_t end_slice_index_exclusive,
    std::size_t worker_index) {
    static_assert(
        std::is_same_v<TensorSlice::ords_t, tt_xy_pair>,
        "generate_slice_sequence_on_dim_v2 not yet implemented for type not of tt_xy_pair");
    // We don't support 4D shapes in the CCL kernels yet, which are needed for proper reduction/concatenation in some
    // cases so for now we subtract the outer dims from the fracture_dim since we only support 2D at the moment.
    if (fracture_dim == 3) {
        fracture_dim -= 2;
    } else {
        // dims are
        fracture_dim = 0;
    }

    TT_ASSERT(worker_slice_shape.y == 1);

    std::vector<TensorSlice> slices;
    auto dim_size = fracture_dim == 1 ? tensor_shape.x : tensor_shape.y;
    TT_ASSERT(dim_size % num_slices == 0);
    auto slice_size_on_dim = dim_size / num_slices;
    auto slice_shape = fracture_dim == 0 ? tt_xy_pair{tensor_shape.x, slice_size_on_dim}
                                         : tt_xy_pair{slice_size_on_dim, tensor_shape.y};

    auto dim_start_offset = start_slice_index * slice_size_on_dim;
    TensorSlice::ords_t tensor_slice_offset =
        fracture_dim == 0 ? tt_xy_pair{0, dim_start_offset} : tt_xy_pair{dim_start_offset, 0};

    bool forward_direction = start_slice_index > end_slice_index_exclusive;  // only for debug
    auto incr = start_slice_index < end_slice_index_exclusive ? 1 : -1;
    if (forward_direction) {
        log_trace(tt::LogOp, "slice_size_on_dim {}", slice_size_on_dim);
        log_trace(tt::LogOp, "worker_index {}", worker_index);
    }

    auto worker_slice_start_offset = worker_slice_offset;

    auto generate_slice = [forward_direction,
                           incr,
                           &slices,
                           &tensor_shape,
                           &slice_shape,
                           &worker_slice_shape,
                           tensor_slice_offset,
                           &worker_slice_start_offset,
                           fracture_dim,
                           dim_start_offset,
                           slice_size_on_dim](std::int64_t i) {
        auto tensor_slice_offset_adjusted = tensor_slice_offset;
        if (fracture_dim == 0) {
            tensor_slice_offset_adjusted.y = slice_size_on_dim * i;
        } else {
            tensor_slice_offset_adjusted.x = slice_size_on_dim * i;
        }
        TT_ASSERT(tensor_shape.x > 0, "Invalid tensor shape. x = 0 but it must be > 0");
        TT_ASSERT(tensor_shape.y > 0, "Invalid tensor shape. y = 0 but it must be > 0");
        TT_ASSERT(slice_shape.x > 0, "Invalid tensor slice shape. x = 0 but it must be > 0");
        TT_ASSERT(slice_shape.y > 0, "Invalid tensor slice shape. x = 0 but it must be > 0");
        TT_ASSERT(
            tensor_slice_offset_adjusted.x < tensor_shape.x,
            "Invalid tensor slice offset. x = {} but it must be < tensor shape x={}. slice_offset: (y={},x={}), "
            "tensor_shape: (y={},x={}). slice_size_on_dim: {}, i: {}",
            tensor_slice_offset_adjusted.x,
            tensor_shape.x,
            tensor_slice_offset_adjusted.y,
            tensor_slice_offset_adjusted.x,
            tensor_shape.y,
            tensor_shape.x,
            slice_size_on_dim,
            i);
        TT_ASSERT(
            tensor_slice_offset_adjusted.y < tensor_shape.y,
            "Invalid tensor slice offset. y = {} but it must be < tensor shape y={}. slice_offset: (y={},x={}), "
            "tensor_shape: (y={},x={}). slice_size_on_dim: {}, i: {}",
            tensor_slice_offset_adjusted.y,
            tensor_shape.y,
            tensor_slice_offset_adjusted.y,
            tensor_slice_offset_adjusted.x,
            tensor_shape.y,
            tensor_shape.x,
            slice_size_on_dim,
            i);
        TT_ASSERT(worker_slice_shape.x > 0, "Invalid worker slice shape. x = 0 but it must be > 0");
        TT_ASSERT(worker_slice_shape.y > 0, "Invalid worker slice shape. y = 0 but it must be > 0");

        const auto& tensor_slice = TensorSlice(
            tensor_shape,
            slice_shape,
            tensor_slice_offset_adjusted,
            worker_slice_shape,
            worker_slice_start_offset,
            fracture_dim);
        if (forward_direction) {
            log_trace(
                tt::LogOp,
                "generate_slice ({}):\n\ttensor_shape: (y={},x={})\n\ttensor_slice_shape: "
                "(y={},x={})\n\ttensor_slice_offset_adjusted: (y={},x={})\n\tslice_start_shape: (y={},x={})\n\tworker "
                "relative slice_start_offset: (y={},x={})\n\tfracture_dim: {}\n\tdim_start_offset: "
                "{}\n\tslice_size_on_dim: {}\n",
                i,
                tensor_slice.tensor_shape.y,
                tensor_slice.tensor_shape.x,
                tensor_slice.tensor_slice_shape.y,
                tensor_slice.tensor_slice_shape.x,
                tensor_slice.tensor_slice_offset.y,
                tensor_slice.tensor_slice_offset.x,
                tensor_slice.worker_slice_shape.y,
                tensor_slice.worker_slice_shape.x,
                tensor_slice.worker_slice_offset.y,
                tensor_slice.worker_slice_offset.x,
                fracture_dim,
                dim_start_offset,
                slice_size_on_dim);
        }

        slices.push_back(tensor_slice);
    };

    for (int i = start_slice_index; i != end_slice_index_exclusive; i += incr) {
        generate_slice(i);
    }

    return slices;
}

GenericWrappedTensorSlicer::GenericWrappedTensorSlicer(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    int slice_dim,
    uint32_t partition_index,
    uint32_t partition_size,
    uint32_t total_num_workers,
    uint32_t max_slice_size_in_bytes,
    uint32_t half_cb_n_pages) {
    this->initialize(
        input_tensor,
        output_tensor,
        slice_dim,
        partition_index,
        partition_size,
        total_num_workers,
        max_slice_size_in_bytes,
        half_cb_n_pages);
}

tt_xy_pair GenericWrappedTensorSlicer::calculate_tensor_slice_shape(
    const Tensor& input_tensor, int slice_dim, uint32_t partition_size) {
    const uint32_t num_tiles_x = input_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    uint32_t num_tiles_y = (input_tensor.padded_shape()[-2] / tt::constants::TILE_HEIGHT);
    for (std::size_t i = 0; input_tensor.padded_shape().rank() > 2 && i < input_tensor.padded_shape().rank() - 2; i++) {
        num_tiles_y *= input_tensor.padded_shape()[i];
    }
    TT_ASSERT(num_tiles_x >= partition_size);
    tt_xy_pair tensor_slice_shape;
    tensor_slice_shape.x = slice_dim == 3 ? (num_tiles_x / partition_size) : num_tiles_x;
    tensor_slice_shape.y = slice_dim != 3 ? num_tiles_y / partition_size : num_tiles_y;
    return tensor_slice_shape;
}

void GenericWrappedTensorSlicer::initialize(
    const Tensor& input_tensor,
    const Tensor& /*output_tensor*/,
    int slice_dim,
    uint32_t partition_index,
    uint32_t partition_size,
    uint32_t total_num_workers,
    uint32_t max_slice_size_in_bytes,
    uint32_t half_cb_n_pages) {
    // Configure layout parameters
    this->row_major = (input_tensor.layout() == Layout::ROW_MAJOR);
    this->input_page_size = input_tensor.buffer()->page_size();
    this->partition_index = partition_index;
    this->partition_size = partition_size;

    // Assume everything in Tile layout for now, row major not supported yet
    TT_FATAL(!this->row_major, "Row major not supported yet");

    this->tensor_slice_shape = calculate_tensor_slice_shape(input_tensor, slice_dim, partition_size);

    // Calculate worker slice shapes (tile layout)
    this->worker_slice_shapes = create_worker_slice_shapes_for_tile_layout(
        input_tensor.padded_shape(),
        this->tensor_slice_shape,
        total_num_workers,
        max_slice_size_in_bytes / this->input_page_size,
        half_cb_n_pages);

    // Flattened tensor shape (tile layout)
    this->flattened_tensor_shape = tt_xy_pair{
        input_tensor.padded_shape()[3] / tt::constants::TILE_WIDTH,
        (input_tensor.padded_shape()[0] * input_tensor.padded_shape()[1] * input_tensor.padded_shape()[2]) /
            tt::constants::TILE_HEIGHT};

    this->worker_slice_offsets = compute_worker_slice_offsets(this->worker_slice_shapes, this->tensor_slice_shape);
}

ccl::InterleavedTensorWorkerSlice GenericWrappedTensorSlicer::get_worker_slice(std::size_t global_worker_index) {
    assert(global_worker_index < this->worker_slice_shapes.size());
    assert(global_worker_index < this->worker_slice_offsets.size());
    return ccl::InterleavedTensorWorkerSlice(
        this->flattened_tensor_shape,
        this->tensor_slice_shape,
        this->worker_slice_shapes[global_worker_index],
        this->worker_slice_offsets[global_worker_index],
        true  // wrapped
    );
}

std::vector<tt_xy_pair> GenericWrappedTensorSlicer::compute_worker_slice_offsets(
    const std::vector<tt_xy_pair>& worker_slice_shapes, const tt_xy_pair& tensor_slice_shape) {
    return compute_worker_slice_offsets_for_wrapped_tensor_slicer(worker_slice_shapes, tensor_slice_shape);
}

std::vector<tt_xy_pair> GenericWrappedTensorSlicer::create_worker_slice_shapes_for_tile_layout(
    const ttnn::Shape& /*tensor_shape*/,
    const tt_xy_pair& tensor_slice_shape_in_tiles,
    uint32_t num_workers,
    uint32_t max_slice_size_in_pages,
    uint32_t /*half_cb_n_pages*/) {
    log_trace(tt::LogOp, "\tmax_slice_size_in_pages={}", max_slice_size_in_pages);
    TT_ASSERT(max_slice_size_in_pages > 0);
    std::vector<tt_xy_pair> worker_slice_shapes;
    worker_slice_shapes.reserve(num_workers);
    const uint32_t total_num_tiles = tensor_slice_shape_in_tiles.x * tensor_slice_shape_in_tiles.y;
    if (num_workers > total_num_tiles) {
        log_warning(
            tt::LogOp,
            "Reduce Scatter more workers instantiated than is work to be done. Some workers will be idle and do "
            "nothing");
        for (uint32_t w = 0; w < total_num_tiles; ++w) {
            worker_slice_shapes.emplace_back(1, 1);
        }
        for (uint32_t w = total_num_tiles; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(0, 0);
        }
        return worker_slice_shapes;
    }

    std::size_t max_slice_size_in_tiles = max_slice_size_in_pages;

    // Assign slices by assuming that the input tensor is flattened into a 1D Shape
    std::size_t optim_worker_slice_len_tiles = std::ceil(
        static_cast<float>(total_num_tiles) /
        num_workers);  // Ceil so that the remainder worker will have a smaller slice

    log_trace(tt::LogOp, "---- GenericWrappedTensorSlicer::create_worker_slice_shapes_for_tile_layout ---- ");
    log_trace(tt::LogOp, "total_num_tiles: {}", total_num_tiles);
    log_trace(tt::LogOp, "num_workers: {}", num_workers);
    log_trace(tt::LogOp, "optim_worker_slice_len_tiles: {}", optim_worker_slice_len_tiles);

    if (max_slice_size_in_tiles < optim_worker_slice_len_tiles) {  // Each worker will have a full slice
        for (uint32_t w = 0; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(max_slice_size_in_tiles, 1);
        }
    } else {  // Each worker will only have one slice
        uint32_t remainder_worker_len_tiles = total_num_tiles % optim_worker_slice_len_tiles;

        for (uint32_t w = 0; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(optim_worker_slice_len_tiles, 1);
        }
        // If there is a remainder worker, we need to adjust the last worker's slice shape to be smaller
        if (remainder_worker_len_tiles > 0) {
            worker_slice_shapes.back() = tt_xy_pair{remainder_worker_len_tiles, 1};
        }
    }

    log_trace(tt::LogOp, "--------------------------------");

    return worker_slice_shapes;
}

GenericWrappedTensorSlicerV2::GenericWrappedTensorSlicerV2(
    const Tensor& input_tensor,
    int slice_dim,
    uint32_t partition_index,
    uint32_t partition_size,
    uint32_t total_num_workers) {
    this->initialize(input_tensor, slice_dim, partition_index, partition_size, total_num_workers);
}

Shape4D<uint32_t> GenericWrappedTensorSlicerV2::calculate_tensor_slice_shape(
    const Shape4D<uint32_t>& input_shape, int slice_dim, uint32_t partition_size) {
    // Calculate the size of the slice along the given dimension
    uint32_t dim_size = input_shape[slice_dim];
    uint32_t slice_size = dim_size / partition_size;

    // Start with full shape
    Shape4D<uint32_t> slice_shape(input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

    TT_FATAL(
        slice_dim >= 0 && slice_dim < 4,
        "Invalid slice dimension. Must be between 0 and 3 but got {}. This should have been normalized to fit within "
        "the range",
        slice_dim);
    slice_shape[slice_dim] = slice_size;

    return slice_shape;
}

Shape4D<uint32_t> GenericWrappedTensorSlicerV2::calculate_tensor_slice_offset(
    const Shape4D<uint32_t>& input_shape, int slice_dim, uint32_t partition_index) const {
    Shape4D<uint32_t> offset(0, 0, 0, 0);

    // Calculate the size of the slice along the given dimension
    uint32_t dim_size = input_shape[slice_dim];
    uint32_t slice_size = dim_size / partition_size;

    TT_FATAL(
        slice_dim >= 0 && slice_dim < 4,
        "Invalid slice dimension. Must be between 0 and 3 but got {}. This should have been normalized to fit within "
        "the range",
        slice_dim);
    offset[slice_dim] = partition_index * slice_size;

    return offset;
}

void GenericWrappedTensorSlicerV2::initialize(
    const Tensor& input_tensor,
    int slice_dim,
    uint32_t partition_index,
    uint32_t partition_size,
    uint32_t total_num_workers) {
    // Configure layout parameters
    this->row_major = (input_tensor.layout() == Layout::ROW_MAJOR);
    this->input_page_size = input_tensor.buffer()->page_size();
    this->partition_index = partition_index;
    this->partition_size = partition_size;

    // Assume everything in Tile layout for now, row major not supported yet
    TT_FATAL(!this->row_major, "Row major not supported yet");

    // Record the input tensor shape
    auto input_shape = input_tensor.padded_shape();
    this->tensor_shape = Shape4D<uint32_t>(
        input_shape[0],
        input_shape[1],
        input_shape[2] / tt::constants::TILE_HEIGHT,
        input_shape[3] / tt::constants::TILE_WIDTH);

    // Calculate tensor slice shape
    this->tensor_slice_shape = calculate_tensor_slice_shape(this->tensor_shape, slice_dim, partition_size);

    // Calculate tensor slice offset
    this->tensor_slice_offset = calculate_tensor_slice_offset(this->tensor_shape, slice_dim, partition_index);

    // Calculate worker slice shapes in terms of flattened tiles
    this->worker_slice_shapes = create_worker_slice_shapes_for_tile_layout(this->tensor_slice_shape, total_num_workers);

    // Calculate worker slice offsets in terms of flattened tiles
    this->worker_slice_offsets = compute_worker_slice_offsets(this->worker_slice_shapes);
}

ttnn::ccl::v2::TensorSlice GenericWrappedTensorSlicerV2::get_worker_slice_v2(std::size_t global_worker_index) {
    assert(global_worker_index < this->worker_slice_shapes.size());
    assert(global_worker_index < this->worker_slice_offsets.size());
    return ttnn::ccl::v2::TensorSlice(
        this->tensor_shape,                              // tensor_shape
        this->tensor_slice_shape,                        // tensor_slice_shape
        this->tensor_slice_offset,                       // tensor_slice_offset
        this->worker_slice_shapes[global_worker_index],  // worker_slice_shape
        this->worker_slice_offsets[global_worker_index]  // worker_slice_offset
    );
}

/* Worker slices and offsets are 4D shapes but flattened to 1D in the last dimension*/

std::vector<Shape4D<uint32_t>> GenericWrappedTensorSlicerV2::compute_worker_slice_offsets(
    const std::vector<Shape4D<uint32_t>>& worker_slice_shapes) {
    Shape4D<uint32_t> offset(0, 0, 0, 0);
    std::vector<Shape4D<uint32_t>> worker_slice_offsets;
    worker_slice_offsets.reserve(worker_slice_shapes.size());
    for (const auto& slice_shape : worker_slice_shapes) {
        worker_slice_offsets.push_back(offset);
        offset.x += slice_shape.x;
    }
    return worker_slice_offsets;
}

std::vector<Shape4D<uint32_t>> GenericWrappedTensorSlicerV2::create_worker_slice_shapes_for_tile_layout(
    const Shape4D<uint32_t>& tensor_slice_shape_in_tiles, uint32_t num_workers) {
    std::vector<Shape4D<uint32_t>> worker_slice_shapes;
    worker_slice_shapes.reserve(num_workers);
    const uint32_t total_num_tiles = tensor_slice_shape_in_tiles.x * tensor_slice_shape_in_tiles.y *
                                     tensor_slice_shape_in_tiles.z * tensor_slice_shape_in_tiles.w;
    if (num_workers > total_num_tiles) {
        log_warning(
            tt::LogOp, "More workers instantiated than is work to be done. Some workers will be idle and do nothing");
        for (uint32_t w = 0; w < total_num_tiles; ++w) {
            worker_slice_shapes.emplace_back(1, 1, 1, 1);
        }
        for (uint32_t w = total_num_tiles; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(0, 0, 0, 0);
        }
        return worker_slice_shapes;
    }

    // Assign slices by assuming that the input tensor is flattened into a 1D Shape
    std::size_t optim_worker_slice_len_tiles = std::ceil(
        static_cast<float>(total_num_tiles) /
        num_workers);  // Ceil so that the remainder worker will have a smaller slice

    log_trace(tt::LogOp, "---- GenericWrappedTensorSlicer::create_worker_slice_shapes_for_tile_layout ---- ");
    log_trace(tt::LogOp, "total_num_tiles: {}", total_num_tiles);
    log_trace(tt::LogOp, "num_workers: {}", num_workers);
    log_trace(tt::LogOp, "optim_worker_slice_len_tiles: {}", optim_worker_slice_len_tiles);

    uint32_t remainder_worker_len_tiles = total_num_tiles % optim_worker_slice_len_tiles;

    for (uint32_t w = 0; w < num_workers; ++w) {
        worker_slice_shapes.emplace_back(Shape4D<uint32_t>(1, 1, 1, optim_worker_slice_len_tiles));
    }
    // If there is a remainder worker, we need to adjust the last worker's slice shape to be smaller
    if (remainder_worker_len_tiles > 0) {
        worker_slice_shapes.back() = Shape4D<uint32_t>(1, 1, 1, remainder_worker_len_tiles);
    }

    log_trace(tt::LogOp, "--------------------------------");

    return worker_slice_shapes;
}

void validate_fabric_2d_dynamic_config() {
    auto physical_mesh_shapes = tt::tt_fabric::get_physical_mesh_shapes();
    TT_FATAL(
        physical_mesh_shapes.size() == 1,
        "Fabric 2D dynamic CCLs expected a single Physical Mesh to be instantiated, but got {} meshes",
        physical_mesh_shapes.size());
    const auto& physical_mesh_shape = physical_mesh_shapes.begin()->second;
    TT_FATAL(
        physical_mesh_shape.dims() == 2,
        "Fabric 2D dynamic CCLs are not supported for mesh shape with more than 2 dimensions");
}

std::tuple<size_t, size_t, bool> get_forward_backward_configuration(
    size_t ring_size, size_t ring_index, Topology topology) {
    // Used for experimentation for optimal perf
    // May be uplifted to an op parameter if needed
    constexpr bool enable_dynamic_alternate = false;
    bool dynamic_alternate = false;
    size_t num_targets_forward = 0;
    size_t num_targets_backward = 0;
    if (topology == Topology::Linear) {
        LineTopology line_topology(ring_size, ring_index);
        num_targets_forward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
        num_targets_backward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);
    } else if (topology == ccl::Topology::Ring) {
        // TODO: Commonize
        num_targets_forward = tt::div_up(ring_size - 1, 2);
        num_targets_backward = ring_size - 1 - num_targets_forward;
        constexpr bool static_alternate = true;
        if constexpr (static_alternate) {
            if (ring_index % 2 == 0) {
                std::swap(num_targets_forward, num_targets_backward);
            }
        }
        if constexpr (enable_dynamic_alternate) {
            // Even ring size will result in uneven fwd/backward distances
            dynamic_alternate = ring_size % 2 == 0;
        }
    }
    return std::make_tuple(num_targets_forward, num_targets_backward, dynamic_alternate);
}

std::tuple<std::array<uint32_t, 2>, std::array<uint32_t, 2>> get_forward_backward_line_unicast_configuration(
    const MeshCoordinate& /*src_device_coord*/,
    const std::optional<MeshCoordinate>& forward_device_coord,
    const std::optional<MeshCoordinate>& backward_device_coord,
    MeshDevice* mesh_device) {
    std::array<uint32_t, 2> forward_args = {};
    std::array<uint32_t, 2> backward_args = {};

    auto fabric_config = tt::tt_fabric::GetFabricConfig();
    if (tt::tt_fabric::is_2d_fabric_config(fabric_config)) {
        validate_fabric_2d_dynamic_config();
        if (forward_device_coord) {
            auto forward_device_fabric_node_id = mesh_device->get_fabric_node_id(forward_device_coord.value());
            forward_args[0] = *forward_device_fabric_node_id.mesh_id;
            forward_args[1] = forward_device_fabric_node_id.chip_id;
        }
        if (backward_device_coord) {
            auto backward_device_fabric_node_id = mesh_device->get_fabric_node_id(backward_device_coord.value());
            backward_args[0] = *backward_device_fabric_node_id.mesh_id;
            backward_args[1] = backward_device_fabric_node_id.chip_id;
        }
    } else if (tt::tt_fabric::is_1d_fabric_config(fabric_config)) {
        if (forward_device_coord) {
            forward_args[0] = 0;  // dst_mesh_id, unused
            forward_args[1] = 1;  // distance_in_hops
        }
        if (backward_device_coord) {
            backward_args[0] = 0;  // dst_mesh_id, unused
            backward_args[1] = 1;  // distance_in_hops
        }
    } else {
        TT_THROW("Unsupported fabric config");
    }
    return std::make_tuple(forward_args, backward_args);
}

std::tuple<uint32_t, uint32_t> get_forward_backward_line_mcast_distance(
    size_t ring_size, size_t ring_index, Topology topology, bool static_alternate) {
    size_t num_targets_forward = 0;
    size_t num_targets_backward = 0;
    if (topology == Topology::Linear) {
        LineTopology line_topology(ring_size, ring_index);
        num_targets_forward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
        num_targets_backward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);
    } else if (topology == ccl::Topology::Ring) {
        // TODO: Commonize
        num_targets_forward = tt::div_up(ring_size - 1, 2);
        num_targets_backward = ring_size - 1 - num_targets_forward;
        if (static_alternate) {
            if (ring_index % 2 == 0) {
                std::swap(num_targets_forward, num_targets_backward);
            }
        }
    }
    return std::make_tuple(num_targets_forward, num_targets_backward);
}

std::tuple<std::array<uint32_t, 6>, std::array<uint32_t, 6>> get_forward_backward_line_mcast_configuration(
    const MeshCoordinate& src_device_coord,
    const std::optional<MeshCoordinate>& forward_device_coord,
    const std::optional<MeshCoordinate>& backward_device_coord,
    uint32_t num_targets_forward,
    uint32_t num_targets_backward,
    MeshDevice* mesh_device) {
    std::array<uint32_t, 6> forward_args = {};
    std::array<uint32_t, 6> backward_args = {};
    // Used for experimentation for optimal perf
    // May be uplifted to an op parameter if needed
    auto fabric_config = tt::tt_fabric::GetFabricConfig();

    if (tt::tt_fabric::is_2d_fabric_config(fabric_config)) {
        validate_fabric_2d_dynamic_config();
        auto src_fabric_node_id = mesh_device->get_fabric_node_id(src_device_coord);
        auto set_mcast_args = [&src_fabric_node_id](
                                  std::array<uint32_t, 6>& args,
                                  const std::optional<MeshCoordinate>& coord,
                                  uint32_t num_targets,
                                  MeshDevice* mesh_device) {
            if (coord) {
                const auto& dev_coord = *coord;
                auto device_fabric_node_id = mesh_device->get_fabric_node_id(dev_coord);
                auto eth_chan_dir =
                    tt::tt_fabric::get_eth_forwarding_direction(src_fabric_node_id, device_fabric_node_id);
                args[0] = *device_fabric_node_id.mesh_id;
                args[1] = device_fabric_node_id.chip_id;
                args[2 + static_cast<std::uint8_t>(eth_chan_dir.value())] = num_targets;
            }
        };
        set_mcast_args(forward_args, forward_device_coord, num_targets_forward, mesh_device);
        set_mcast_args(backward_args, backward_device_coord, num_targets_backward, mesh_device);
    } else if (tt::tt_fabric::is_1d_fabric_config(fabric_config)) {
        if (forward_device_coord) {
            forward_args[0] = 1;                    // start_distance_in_hops
            forward_args[1] = num_targets_forward;  // range_hops
        }
        if (backward_device_coord) {
            backward_args[0] = 1;                     // start_distance_in_hops
            backward_args[1] = num_targets_backward;  // range_hops
        }
    } else {
        TT_THROW("Unsupported fabric config");
    }
    return std::make_tuple(forward_args, backward_args);
}

void fabric_mux_connection_ct_args(
    const uint32_t num_workers_per_direction,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args) {
    worker_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));  // fabric_mux_num_buffers_per_channel 0
    worker_ct_args.push_back(
        mux_kernel_config.get_buffer_size_bytes(channel_type));        // fabric_mux_channel_buffer_size_bytes 1
    worker_ct_args.push_back(mux_kernel_config.get_status_address());  // fabric_mux_status_address 2
    worker_ct_args.push_back(
        mux_kernel_config.get_termination_signal_address());  // fabric_mux_termination_signal_address 3
    worker_ct_args.push_back(num_workers_per_direction);      // num_mux_clients 4
}

void fabric_mux_connection_rt_args(
    const bool mux_connection_valid,
    const bool is_termination_master,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    std::vector<uint32_t>& worker_rt_args,
    std::optional<uint32_t> termination_master_semaphore_id) {
    worker_rt_args.push_back(mux_connection_valid);   // mux_connection_valid 0
    worker_rt_args.push_back(is_termination_master);  // is_termination_master 1
    worker_rt_args.push_back(mux_virtual_core.x);     // fabric_mux_x 2
    worker_rt_args.push_back(mux_virtual_core.y);     // fabric_mux_y 3
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_base_address(channel_type, worker_id));  // fabric_mux_channel_base_address 4
    worker_rt_args.push_back(mux_kernel_config.get_connection_info_address(
        channel_type, worker_id));  // fabric_mux_connection_info_address 5
    worker_rt_args.push_back(mux_kernel_config.get_connection_handshake_address(
        channel_type, worker_id));  // fabric_mux_connection_handshake_address 6
    worker_rt_args.push_back(
        mux_kernel_config.get_flow_control_address(channel_type, worker_id));  // fabric_mux_flow_control_address 7
    worker_rt_args.push_back(
        mux_kernel_config.get_buffer_index_address(channel_type, worker_id));  // fabric_mux_buffer_index_address 8
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));  // fabric_mux_channel_id 9
    worker_rt_args.push_back(termination_master_semaphore_id.value_or(
        CreateSemaphore(program, {worker_logical_core}, 0)));                      // termination_sync_address 10
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_fabric_mux_status_address 11
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_flow_control_address 12
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_teardown_address 13
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_buffer_index_address 14
    worker_rt_args.push_back(termination_master_virtual_core.x);                   // termination_master_noc_x 15
    worker_rt_args.push_back(termination_master_virtual_core.y);                   // termination_master_noc_y 16
}

namespace {  // anonymous namespace — internal helpers for Fabric perf model

// ==================== Fabric Perf Model Helpers ====================
//
// Measured fabric bandwidth and latency data sourced from:
//   tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden/*.csv
//
// When no exact match is found the lookup functions perform an estimation based
// on available data points and log a warning.

using ClusterType = tt::tt_metal::ClusterType;
using Topology = tt::tt_fabric::Topology;

// ---------- BW type aliases ----------
// BwMap: (cluster, noc_type) -> packet_size -> num_links -> bw_gbps
using LinkBwMap = std::map<uint32_t, float>;     // num_links -> GB/s
using PktBwMap = std::map<uint32_t, LinkBwMap>;  // packet_size -> LinkBwMap
using BwKey = std::pair<ClusterType, FabricNocWriteType>;
using BwMap = std::map<BwKey, PktBwMap>;

// ---------- Bandwidth tables (one per FabricWriteType) ----------
// Source: tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden/
// Data: avg_bandwidth_gigabytes_per_s column.

static const BwMap unicast_bw = {
    {{ClusterType::T3K, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 8.330187f}}},
         {4096, {{1, 9.485218f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 7.152492f}}},
         {4096, {{1, 9.879004f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 3.465731f}}},
         {4096, {{1, 6.038864f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 7.285941f}, {2, 6.183416f}, {3, 6.180227f}, {4, 6.178862f}}},
         {4096, {{1, 7.198969f}, {2, 7.035704f}, {3, 6.996289f}, {4, 6.942920f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 7.130254f}, {2, 7.126786f}, {3, 7.127653f}, {4, 7.127830f}}},
         {4096, {{1, 7.220976f}, {2, 7.062391f}, {3, 7.029766f}, {4, 6.973647f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 3.437590f}, {2, 3.440290f}, {3, 3.445589f}, {4, 3.386013f}}},
         {4096, {{1, 5.827964f}, {2, 5.854786f}, {3, 5.876048f}, {4, 5.494558f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 19.228799f}, {2, 19.218864f}, {3, 19.223229f}, {4, 19.225030f}}},
         {4096, {{1, 38.378649f}, {2, 38.362487f}, {3, 38.386210f}, {4, 38.375373f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 19.097741f}, {2, 19.061249f}, {3, 19.046550f}, {4, 19.065744f}}},
         {4096, {{1, 36.977753f}, {2, 36.990636f}, {3, 36.975830f}, {4, 36.886060f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 5.886736f}, {2, 5.885360f}, {3, 5.886119f}, {4, 5.885907f}}},
         {4096, {{1, 10.928358f}, {2, 10.931768f}, {3, 10.926167f}, {4, 10.928504f}}},
     }},
};

static const BwMap linear_mcast_bw = {
    {{ClusterType::T3K, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 6.832798f}}},
         {3264, {{1, 9.974803f}}},
         {4096, {{1, 11.047054f}}},
         {5440, {{1, 11.546920f}}},
         {7616, {{1, 10.608683f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 5.867437f}}},
         {4096, {{1, 10.532368f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 3.467653f}}},
         {4096, {{1, 5.818009f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 5.844904f}, {2, 5.843234f}, {3, 5.842464f}, {4, 5.842379f}}},
         {3264, {{1, 8.966020f}, {2, 8.963147f}, {3, 8.963453f}, {4, 8.962912f}}},
         {4096, {{1, 10.229066f}, {2, 10.152595f}, {3, 10.152880f}, {4, 10.112945f}}},
         {5440, {{1, 10.661632f}, {2, 10.615237f}, {3, 10.492308f}, {4, 10.619480f}}},
         {7616, {{1, 9.197206f}, {2, 9.014086f}, {3, 8.905171f}, {4, 8.702418f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 5.118836f}, {2, 5.114608f}, {3, 5.114015f}, {4, 5.113653f}}},
         {4096, {{1, 9.688844f}, {2, 9.657664f}, {3, 9.658564f}, {4, 9.659046f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 3.226826f}, {2, 2.894843f}, {3, 2.896755f}, {4, 2.938657f}}},
         {4096, {{1, 5.524707f}, {2, 5.071252f}, {3, 4.997584f}, {4, 4.947952f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 14.836414f}, {2, 14.824661f}, {3, 14.835477f}, {4, 14.831378f}}},
         {3264, {{1, 23.661143f}, {2, 23.665936f}, {3, 23.666386f}, {4, 23.661644f}}},
         {4096, {{1, 29.067834f}, {2, 29.066246f}, {3, 29.030834f}, {4, 29.039209f}}},
         {5440, {{1, 38.171438f}, {2, 38.143406f}, {3, 38.160707f}, {4, 38.152969f}}},
         {7616, {{1, 37.510592f}, {2, 37.489599f}, {3, 37.487505f}, {4, 37.488191f}}},
         {8704, {{1, 41.047545f}, {2, 41.021882f}, {3, 41.031321f}, {4, 41.021445f}}},
         {15232, {{1, 41.841001f}, {2, 41.678271f}, {3, 41.544213f}, {4, 40.689124f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 14.392820f}, {2, 14.390522f}, {3, 14.387979f}, {4, 14.387445f}}},
         {4096, {{1, 28.821150f}, {2, 28.809243f}, {3, 28.813908f}, {4, 28.815657f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 5.419047f}, {2, 5.418843f}, {3, 5.417605f}, {4, 5.418696f}}},
         {4096, {{1, 10.126003f}, {2, 10.125870f}, {3, 10.125482f}, {4, 10.124974f}}},
     }},
};

static const BwMap full_ring_mcast_bw = {
    {{ClusterType::T3K, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 4.516946f}}},
         {4096, {{1, 8.924658f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 3.968142f}}},
         {4096, {{1, 7.926720f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 2.453754f}}},
         {4096, {{1, 4.201091f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 4.274430f}, {2, 4.266709f}, {3, 4.245975f}, {4, 4.265941f}}},
         {4096, {{1, 8.389139f}, {2, 8.359459f}, {3, 8.355918f}, {4, 8.034062f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 3.860096f}, {2, 3.858300f}, {3, 3.858585f}, {4, 3.857108f}}},
         {4096, {{1, 7.654638f}, {2, 7.649279f}, {3, 7.650576f}, {4, 7.608103f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 2.448776f}, {2, 2.300975f}, {3, 2.281434f}, {4, 2.222657f}}},
         {4096, {{1, 4.144041f}, {2, 4.083543f}, {3, 4.099715f}, {4, 4.034203f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 8.922464f}, {2, 8.957256f}, {3, 8.918135f}, {4, 8.888296f}}},
         {4096, {{1, 18.054548f}, {2, 17.994120f}, {3, 17.981628f}, {4, 18.091492f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 7.388885f}, {2, 7.306202f}, {3, 7.336794f}, {4, 7.305242f}}},
         {4096, {{1, 15.146433f}, {2, 14.815784f}, {3, 14.922098f}, {4, 15.013104f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 3.612803f}, {2, 3.601608f}, {3, 3.601758f}, {4, 3.599812f}}},
         {4096, {{1, 6.821457f}, {2, 6.800818f}, {3, 6.757227f}, {4, 6.698880f}}},
     }},
};

static const BwMap half_ring_mcast_bw = {
    {{ClusterType::T3K, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 4.689026f}}},
         {4096, {{1, 9.464704f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 4.136056f}}},
         {4096, {{1, 8.224739f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 2.653695f}}},
         {4096, {{1, 4.563792f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 4.427287f}, {2, 4.381615f}, {3, 4.380846f}, {4, 4.396641f}}},
         {4096, {{1, 8.947899f}, {2, 8.985732f}, {3, 8.852474f}, {4, 8.725974f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 3.873709f}, {2, 3.898142f}, {3, 3.897968f}, {4, 3.885799f}}},
         {4096, {{1, 8.301195f}, {2, 8.318888f}, {3, 8.305937f}, {4, 8.269113f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 2.490845f}, {2, 2.326145f}, {3, 2.279853f}, {4, 2.293614f}}},
         {4096, {{1, 4.146097f}, {2, 4.108749f}, {3, 4.140277f}, {4, 4.080597f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 15.708364f}, {2, 15.708924f}, {3, 15.709515f}, {4, 15.611534f}}},
         {4096, {{1, 31.380088f}, {2, 31.381955f}, {3, 31.190107f}, {4, 31.008662f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 12.212097f}, {2, 12.212180f}, {3, 12.211827f}, {4, 12.212270f}}},
         {4096, {{1, 24.406643f}, {2, 24.407469f}, {3, 24.405264f}, {4, 24.350952f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 4.788901f}, {2, 4.785855f}, {3, 4.785788f}, {4, 4.785755f}}},
         {4096, {{1, 8.998680f}, {2, 8.983877f}, {3, 8.982882f}, {4, 8.982102f}}},
     }},
};

static const BwMap mesh_mcast_bw = {
    {{ClusterType::T3K, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 4.015804f}}},
         {3264, {{1, 7.007417f}}},
         {4096, {{1, 7.429800f}}},
         {5440, {{1, 8.706840f}}},
         {7616, {{1, 6.179132f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 3.770821f}}},
         {4096, {{1, 7.008948f}}},
     }},
    {{ClusterType::T3K, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 2.901026f}}},
         {4096, {{1, 5.155640f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 3.643313f}, {2, 3.639977f}, {3, 3.616495f}, {4, 3.615987f}}},
         {3264, {{1, 5.805356f}, {2, 5.802861f}, {3, 5.759248f}, {4, 5.757365f}}},
         {4096, {{1, 7.279113f}, {2, 7.247523f}, {3, 7.214281f}, {4, 7.205492f}}},
         {5440, {{1, 6.901538f}, {2, 6.640762f}, {3, 6.619454f}, {4, 6.458215f}}},
         {7616, {{1, 7.379932f}, {2, 7.212734f}, {3, 7.197460f}, {4, 7.027609f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 3.294091f}, {2, 3.287417f}, {3, 3.283250f}, {4, 3.280403f}}},
         {4096, {{1, 6.502396f}, {2, 6.475077f}, {3, 6.470928f}, {4, 6.467910f}}},
     }},
    {{ClusterType::GALAXY, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 2.049108f}, {2, 2.033526f}, {3, 2.029798f}, {4, 2.023453f}}},
         {4096, {{1, 3.721428f}, {2, 3.650082f}, {3, 3.667560f}, {4, 3.640862f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::UnicastWrite},
     {
         {2048, {{1, 13.147748f}, {2, 13.156140f}, {3, 13.152273f}, {4, 13.148448f}}},
         {3264, {{1, 20.985502f}, {2, 20.974462f}, {3, 20.972590f}, {4, 20.970967f}}},
         {4096, {{1, 26.454910f}, {2, 26.455543f}, {3, 26.469075f}, {4, 26.447972f}}},
         {5440, {{1, 35.276519f}, {2, 35.259502f}, {3, 35.271609f}, {4, 33.666475f}}},
         {7616, {{1, 42.628063f}, {2, 42.689319f}, {3, 42.579568f}, {4, 31.162960f}}},
         {8704, {{1, 45.489619f}, {2, 45.557583f}, {3, 42.971700f}, {4, 31.270416f}}},
         {15232, {{1, 36.955361f}, {2, 36.899463f}, {3, 37.150128f}, {4, 31.376570f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::ScatterWrite},
     {
         {2048, {{1, 13.293138f}, {2, 13.294820f}, {3, 13.291380f}, {4, 13.292832f}}},
         {4096, {{1, 26.466289f}, {2, 26.454236f}, {3, 26.444455f}, {4, 26.455345f}}},
     }},
    {{ClusterType::P150_X4, FabricNocWriteType::FusedAtomicInc},
     {
         {2048, {{1, 5.242205f}, {2, 5.234469f}, {3, 5.226773f}, {4, 5.215834f}}},
         {4096, {{1, 9.796307f}, {2, 9.778249f}, {3, 9.754941f}, {4, 9.718332f}}},
     }},
};

// ---------- Per-hop latency ----------
// Source: tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden/
// Data: payload=4096B, per_hop_avg_ns column.
using Topology = tt::tt_fabric::Topology;
using LatencyKey = std::tuple<ClusterType, Topology, FabricNocWriteType>;
static const std::map<LatencyKey, float> fabric_hop_latency = {
    // T3K — golden_latency_summary_wormhole_b0_t3k.csv
    {{ClusterType::T3K, Topology::Linear, FabricNocWriteType::UnicastWrite}, 1442.52f},
    {{ClusterType::T3K, Topology::Linear, FabricNocWriteType::FusedAtomicInc}, 1468.02f},
    {{ClusterType::T3K, Topology::Ring, FabricNocWriteType::UnicastWrite}, 1480.62f},
    {{ClusterType::T3K, Topology::Ring, FabricNocWriteType::FusedAtomicInc}, 1516.39f},
    {{ClusterType::T3K, Topology::Mesh, FabricNocWriteType::UnicastWrite}, 1547.46f},
    {{ClusterType::T3K, Topology::Mesh, FabricNocWriteType::FusedAtomicInc}, 1583.55f},
    // GALAXY — golden_latency_summary_wormhole_b0_galaxy.csv
    {{ClusterType::GALAXY, Topology::Linear, FabricNocWriteType::UnicastWrite}, 1513.12f},
    {{ClusterType::GALAXY, Topology::Linear, FabricNocWriteType::FusedAtomicInc}, 1455.37f},
    {{ClusterType::GALAXY, Topology::Ring, FabricNocWriteType::UnicastWrite}, 1557.76f},
    {{ClusterType::GALAXY, Topology::Ring, FabricNocWriteType::FusedAtomicInc}, 1547.89f},
    {{ClusterType::GALAXY, Topology::Mesh, FabricNocWriteType::UnicastWrite}, 1592.02f},
    {{ClusterType::GALAXY, Topology::Mesh, FabricNocWriteType::FusedAtomicInc}, 1638.07f},
    // P150_X4 — golden_latency_summary_blackhole_p150_x4.csv
    {{ClusterType::P150_X4, Topology::Linear, FabricNocWriteType::UnicastWrite}, 819.04f},
    {{ClusterType::P150_X4, Topology::Linear, FabricNocWriteType::FusedAtomicInc}, 822.14f},
    {{ClusterType::P150_X4, Topology::Ring, FabricNocWriteType::UnicastWrite}, 867.46f},
    {{ClusterType::P150_X4, Topology::Ring, FabricNocWriteType::FusedAtomicInc}, 865.18f},
    {{ClusterType::P150_X4, Topology::Mesh, FabricNocWriteType::UnicastWrite}, 906.47f},
    {{ClusterType::P150_X4, Topology::Mesh, FabricNocWriteType::FusedAtomicInc}, 876.62f},
};

// ---------- Helpers ----------

// Map unknown ClusterTypes to the nearest cluster with measured data.
ClusterType normalize_cluster(ClusterType ct) {
    switch (ct) {
        case ClusterType::T3K:
        case ClusterType::GALAXY:
        case ClusterType::P150_X4: return ct;
        case ClusterType::P150:
        case ClusterType::P150_X2:
        case ClusterType::P150_X8:
        case ClusterType::BLACKHOLE_GALAXY:
            log_warning(
                tt::LogOp,
                "Fabric perf model: no data for ClusterType {}; using P150_X4 as estimate.",
                static_cast<int>(ct));
            return ClusterType::P150_X4;
        default:
            log_warning(
                tt::LogOp, "Fabric perf model: unknown ClusterType {}; using T3K as estimate.", static_cast<int>(ct));
            return ClusterType::T3K;
    }
}

// Map topology variants to the canonical ones with measured data.
Topology normalize_topology(Topology topo) {
    switch (topo) {
        case Topology::Linear:
        case Topology::Ring:
        case Topology::Mesh: return topo;
        case Topology::Torus: return Topology::Ring;
        case Topology::NeighborExchange:
        default: return Topology::Linear;
    }
}

const BwMap& get_bw_table(FabricWriteType wt) {
    switch (wt) {
        case FabricWriteType::Unicast: return unicast_bw;
        case FabricWriteType::MulticastLinear: return linear_mcast_bw;
        case FabricWriteType::MulticastFullRing: return full_ring_mcast_bw;
        case FabricWriteType::MulticastHalfRing: return half_ring_mcast_bw;
        case FabricWriteType::MulticastMesh: return mesh_mcast_bw;
        default: return linear_mcast_bw;
    }
}

// Snap to nearest key in a sorted map.
float snap_nearest(const std::map<uint32_t, float>& m, uint32_t key) {
    TT_ASSERT(!m.empty());
    auto hi = m.lower_bound(key);
    if (hi != m.end() && hi->first == key) {
        return hi->second;
    }
    if (hi == m.end()) {
        return std::prev(hi)->second;
    }
    if (hi == m.begin()) {
        return hi->second;
    }
    auto lo = std::prev(hi);
    return (key - lo->first <= hi->first - key) ? lo->second : hi->second;
}

// Piecewise-linear interpolation on a sorted map<uint32_t, float>.
// Exact key → exact value.  Between two keys → lerp.
// Below min → extrapolate from two lowest (clamped to 0).
// Above max → clamp to highest measured value.
float lerp_map(const std::map<uint32_t, float>& m, uint32_t key) {
    TT_ASSERT(!m.empty());
    auto hi = m.lower_bound(key);
    if (hi != m.end() && hi->first == key) {
        return hi->second;  // exact
    }
    if (hi == m.end()) {
        return std::prev(hi)->second;  // above max → clamp
    }
    if (hi == m.begin()) {  // below min → extrapolate (or clamp if single entry)
        if (m.size() == 1) {
            return hi->second;
        }
        auto hi2 = std::next(hi);
        float t = float(int32_t(key) - int32_t(hi->first)) / float(hi2->first - hi->first);
        return std::max(0.0f, std::lerp(hi->second, hi2->second, t));
    }
    // between two keys → lerp
    auto lo = std::prev(hi);
    float t = float(key - lo->first) / float(hi->first - lo->first);
    return std::lerp(lo->second, hi->second, t);
}

// ---------- BW Lookup ----------
float lookup_fabric_bw(
    FabricWriteType write_type,
    FabricNocWriteType noc_type,
    tt::tt_metal::ClusterType cluster,
    uint32_t num_links,
    uint32_t packet_size) {
    const auto& table = get_bw_table(write_type);
    auto nc = normalize_cluster(cluster);

    // Step 1: find (cluster, noc_type) slice
    auto slice_it = table.find({nc, noc_type});
    if (slice_it == table.end()) {
        if (noc_type != FabricNocWriteType::UnicastWrite) {
            log_warning(
                tt::LogOp,
                "Fabric BW: no data for noc_type={}; using UnicastWrite as estimate.",
                static_cast<int>(noc_type));
            slice_it = table.find({nc, FabricNocWriteType::UnicastWrite});
        }
        if (slice_it == table.end()) {
            log_warning(
                tt::LogOp, "Fabric BW: no data for cluster={}; using 1.0 GB/s as fallback.", static_cast<int>(nc));
            return 1.0f;
        }
    }
    const auto& pkt_map = slice_it->second;

    // Step 2: snap num_links (nearest) at each packet_size, then interpolate packet_size (if missing data)
    bool links_estimated = false;
    std::map<uint32_t, float> bw_by_pkt;
    for (const auto& [pkt, link_map] : pkt_map) {
        auto it = link_map.find(num_links);
        if (it != link_map.end()) {
            bw_by_pkt[pkt] = it->second;
        } else {
            bw_by_pkt[pkt] = snap_nearest(link_map, num_links);
            links_estimated = true;
        }
    }
    if (links_estimated) {
        log_warning(
            tt::LogOp, "Fabric BW: no data for num_links={}; snapped to nearest measured link count.", num_links);
    }

    float result = lerp_map(bw_by_pkt, packet_size);
    bool pkt_estimated = (bw_by_pkt.find(packet_size) == bw_by_pkt.end());
    if (pkt_estimated) {
        log_warning(
            tt::LogOp,
            "Fabric BW: no data for packet_size={}; interpolated between neighboring measurements.",
            packet_size);
    }

    return result;
}

// ---------- Latency Lookup ----------
float lookup_fabric_hop_latency_ns(ClusterType cluster, Topology topo, FabricNocWriteType noc_type) {
    auto nc = normalize_cluster(cluster);
    auto nt = normalize_topology(topo);

    auto it = fabric_hop_latency.find({nc, nt, noc_type});
    if (it != fabric_hop_latency.end()) {
        return it->second;
    }

    // ScatterWrite not in latency CSVs — fall back to UnicastWrite
    if (noc_type != FabricNocWriteType::UnicastWrite) {
        log_warning(
            tt::LogOp,
            "Fabric latency: no data for noc_type={}; using UnicastWrite as estimate.",
            static_cast<int>(noc_type));
        it = fabric_hop_latency.find({nc, nt, FabricNocWriteType::UnicastWrite});
        if (it != fabric_hop_latency.end()) {
            return it->second;
        }
    }

    log_warning(tt::LogOp, "Fabric latency: no data found; using 1500 ns/hop as fallback.");
    return 1500.0f;
}

}  // anonymous namespace

float estimate_fabric_transfer_ns(
    int64_t data_bytes,
    uint32_t num_links,
    uint32_t packet_size,
    FabricWriteType write_type,
    FabricNocWriteType noc_type,
    uint32_t num_hops) {
    const auto cluster_type = tt::tt_metal::GetClusterType();
    const float bw_per_link = lookup_fabric_bw(write_type, noc_type, cluster_type, num_links, packet_size);
    const float total_bw = bw_per_link * static_cast<float>(num_links);
    const float transfer_ns = (total_bw > 0.0f) ? static_cast<float>(data_bytes) / total_bw : 0.0f;

    const auto fabric_topo = tt::tt_fabric::get_fabric_topology();
    const float hop_lat = lookup_fabric_hop_latency_ns(cluster_type, fabric_topo, noc_type);
    const float latency_ns = static_cast<float>(num_hops) * hop_lat;

    return transfer_ns + latency_ns;
}

}  // namespace ttnn::ccl
