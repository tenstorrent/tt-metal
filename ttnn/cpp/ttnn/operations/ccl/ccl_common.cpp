// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/ccl_common.hpp"

#include <cstdint>
#include <cmath>

#include "ccl_host_datastructures.hpp"
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"

#include "tt-metalium/hal.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace ccl {

void SyncModeSpec::add_signal(uint32_t sem_id, uint32_t wait_count) {
    this->sem_ids.push_back(sem_id);
    this->wait_counts.push_back(wait_count);
    this->num_signals++;
}

LineTopology::LineTopology(size_t line_size, size_t line_index) : _line_size(line_size), _line_index(line_index) {}

bool LineTopology::is_first_device_in_line(ttnn::ccl::LineDirection direction) const {
    if (direction == ttnn::ccl::LineDirection::FORWARD) {
        return _line_index == 0;
    } else {
        TT_ASSERT(direction == ttnn::ccl::LineDirection::BACKWARD);
        return _line_index == _line_size - 1;
    }
}
bool LineTopology::is_last_device_in_line(ttnn::ccl::LineDirection direction) const {
    if (direction == ttnn::ccl::LineDirection::BACKWARD) {
        return _line_index == 0;
    } else {
        TT_ASSERT(direction == ttnn::ccl::LineDirection::FORWARD);
        return _line_index == _line_size - 1;
    }
}

bool LineTopology::is_at_end_of_line() const { return _line_index == 0 || _line_index == _line_size - 1; }

size_t LineTopology::line_size() const { return _line_size; }

size_t LineTopology::line_index() const { return _line_index; }

size_t LineTopology::get_distance_to_end_of_line(ttnn::ccl::LineDirection direction) const {
    if (direction == ttnn::ccl::LineDirection::FORWARD) {
        return (_line_size - _line_index) - 1;
    } else {
        return _line_index;
    }
}

ttnn::ccl::Topology LineTopology::topology() const { return ttnn::ccl::Topology::Linear; }

tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload_from_programs(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const std::function<tt::tt_metal::operation::ProgramWithCallbacks(const ttnn::MeshCoordinate&)>& create_program) {
    tt::tt_metal::operation::MeshWorkloadWithCallbacks workload_with_callbacks;
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            const ttnn::MeshCoordinateRange program_range(coord, coord);
            auto program_with_callbacks = create_program(coord);
            workload_with_callbacks.workload.add_program(program_range, std::move(program_with_callbacks.program));
            if (program_with_callbacks.override_runtime_arguments_callback.has_value()) {
                workload_with_callbacks.per_program_callbacks.emplace(
                    program_range, std::move(*program_with_callbacks.override_runtime_arguments_callback));
            }
        }
    }
    return workload_with_callbacks;
}

SenderRecieverConfig get_device_sender_receiver_config(
    const IDevice* target_device, const std::vector<IDevice*>& devices, ttnn::ccl::Topology topology) {
    uint32_t num_devices = devices.size();
    bool is_linear = topology == ttnn::ccl::Topology::Linear;
    SenderRecieverConfig config;
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == target_device) {
            config.device_index = i;
            bool is_last_chip_in_clockwise_direction = is_linear && i == (num_devices - 1);
            bool is_last_chip_in_counter_clockwise_direction = is_linear && i == 0;

            config.receiver_device_id = is_last_chip_in_clockwise_direction
                                            ? std::nullopt
                                            : std::optional<chip_id_t>(devices.at((i + 1) % num_devices)->id());

            config.sender_device_id =
                is_last_chip_in_counter_clockwise_direction
                    ? std::nullopt
                    : std::optional<chip_id_t>(devices.at((i + num_devices - 1) % num_devices)->id());
        }
    }

    return config;
}

SenderRecieverConfig get_device_sender_receiver_config_in_ring(
    const MeshCoordinate& mesh_coord,
    const distributed::MeshDevice* mesh_device,
    uint32_t cluster_axis,
    int ring_size) {
    SenderRecieverConfig config;
    const auto& mesh_view = mesh_device->get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "CLL operation invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    const auto view_index = (cluster_axis == 0) ? mesh_coord[1] : mesh_coord[0];
    config.device_index = (cluster_axis == 0) ? mesh_coord[0] : mesh_coord[1];

    auto get_chip_id = [&](std::size_t line_index) -> std::optional<chip_id_t> {
        auto new_row = mesh_coord[0];
        auto new_col = mesh_coord[1];
        if (cluster_axis == 0) {
            new_row = line_index % ring_size;
        } else {
            new_col = line_index % ring_size;
        }
        return mesh_view.find_device_id(MeshCoordinate(new_row, new_col));
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
    auto mesh_device = tensor.mesh_device();
    std::vector<IDevice*> devices = {};
    devices.reserve(tensor.device_storage().coords.size());
    for (const auto& coord : tensor.device_storage().coords) {
        devices.push_back(mesh_device->get_device(coord));
    }
    return devices;
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
    IDevice const* device,
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
            auto const& sockets = device->get_ethernet_sockets(receiver_device);
            auto eth_sender_core = sockets.at(sender_socket_idx);
            eth_sender_cores.push_back(eth_sender_core);
            log_trace(tt::LogOp, "\teth_sender_core on link {}: (x={},y={})", l, eth_sender_core.x, eth_sender_core.y);
        }
        if (!is_linear || ring_index != 0) {
            uint32_t sender_device = sender_device_id.value();
            auto const& sockets = device->get_ethernet_sockets(sender_device);
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

CclOpInterleavedTensorConfig::CclOpInterleavedTensorConfig(Tensor const& input_tensor) :
    CclOpTensorConfig(input_tensor) {}

CclOpShardedTensorConfig::CclOpShardedTensorConfig(Tensor const& tensor) :
    CclOpTensorConfig(tensor), shard_spec(tensor.shard_spec().value()) {}

const tt::tt_metal::ShardSpec& CclOpShardedTensorConfig::get_shard_spec() const { return this->shard_spec; }

std::unique_ptr<CclOpTensorConfig> CclOpTensorConfig::build_all_gather_tensor_config(Tensor const& tensor) {
    if (tensor.is_sharded()) {
        return std::make_unique<CclOpShardedTensorConfig>(tensor);
    } else {
        return std::make_unique<CclOpInterleavedTensorConfig>(tensor);
    }
}

void generate_edm_kernels_for_ring_or_linear_topology(
    Program& program,
    const IDevice* device,
    const RingTopology& topology_config,
    const std::vector<ccl::EriscDatamoverBuilder>& clockwise_edm_builders,
    const std::vector<ccl::EriscDatamoverBuilder>& counter_clockwise_edm_builders,
    std::optional<uint32_t> receiver_device_id,
    std::optional<uint32_t> sender_device_id) {
    auto sender_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(tt::tt_metal::hal::get_arch());
    auto receiver_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(tt::tt_metal::hal::get_arch());
    for (uint32_t i = 0; i < topology_config.num_links; ++i) {
        bool is_clockwise_direction_edm_enabled =
            !topology_config.is_linear || topology_config.ring_index != topology_config.ring_size - 1;
        if (is_clockwise_direction_edm_enabled) {
            auto eth_sender_core = topology_config.eth_sender_cores.at(i);
            log_trace(tt::LogOp, "EDM CLOCKWISE KERNEL RT ARGS: ");
            auto eth_sender_kernel = generate_edm_kernel(
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
            auto eth_receiver_kernel = generate_edm_kernel(
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

template <typename EDMBuilder>
tt::tt_metal::KernelHandle generate_edm_kernel_impl(
    Program& program,
    const IDevice* device,
    const EDMBuilder& edm_builder,
    const std::string& kernel_path,
    const CoreCoord& eth_core,
    tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id,
    std::optional<tt::tt_metal::KernelBuildOptLevel> opt_level = std::nullopt) {
    edm_builder.dump_to_log();

    std::vector<uint32_t> const edm_kernel_rt_args = edm_builder.get_runtime_args();
    // Ethernet Kernels
    const std::vector<uint32_t> eth_sender_ct_args = edm_builder.get_compile_time_args((uint32_t)risc_id);
    log_trace(tt::LogOp, "EDM core (x={},y={}):", eth_core.x, eth_core.y);
    log_trace(tt::LogOp, "CT ARGS:");
    for (auto const& s : eth_sender_ct_args) {
        log_trace(tt::LogOp, "\t{}", s);
    }

    auto kernel_config =
        tt::tt_metal::EthernetConfig{.noc = noc_id, .processor = risc_id, .compile_args = eth_sender_ct_args};
    if (opt_level.has_value()) {
        kernel_config.opt_level = opt_level.value();
    }
    auto eth_sender_kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        eth_core,
        kernel_config);

    tt::tt_metal::SetRuntimeArgs(program, eth_sender_kernel, eth_core, edm_kernel_rt_args);

    std::stringstream ss;
    ss << "EDM ARGS:\n";
    for (auto const& s : edm_kernel_rt_args) {
        ss << "\t" << s << "\n";
    }
    log_trace(tt::LogOp, "{}", ss.str());

    return eth_sender_kernel;
}

tt::tt_metal::KernelHandle generate_edm_kernel(
    Program& program,
    const IDevice* device,
    const tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core,
    const tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id) {
    return generate_edm_kernel_impl(
        program,
        device,
        edm_builder,
        "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_datamover.cpp",
        eth_core,
        risc_id,
        noc_id,
        tt::tt_metal::KernelBuildOptLevel::O3);
}

tt::tt_metal::KernelHandle generate_edm_kernel(
    Program& program,
    const IDevice* device,
    const ccl::EriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core,
    const tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id) {
    return generate_edm_kernel_impl(
        program,
        device,
        edm_builder,
        "ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp",
        eth_core,
        risc_id,
        noc_id);
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
    const uint32_t channel_buffer_size =
        config.compute_buffer_size(num_channels, num_buffers_per_channel, page_size);
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
    Tensor const& input_tensor,
    Tensor const& output_tensor,
    int slice_dim,
    uint32_t ring_index,
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
    uint32_t slice_size_in_bytes = std::numeric_limits<uint32_t>::max();
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
    Tensor const& input_tensor,
    Tensor const& output_tensor,
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
        half_cb_n_pages){};

RingReduceScatterWrappedTensorSlicer::RingReduceScatterWrappedTensorSlicer(
    Tensor const& input_tensor,
    Tensor const& output_tensor,
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
        half_cb_n_pages){};

std::vector<tt_xy_pair> RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
    std::vector<tt_xy_pair> const& worker_slice_shapes, tt_xy_pair const& tensor_slice_shape) {
    std::vector<tt_xy_pair> worker_slice_offsets;
    worker_slice_offsets.reserve(worker_slice_shapes.size());

    std::size_t offset_x = 0;
    std::size_t offset_y = 0;
    std::size_t last_worker_size_y = worker_slice_shapes.at(0).y;  // for validation
    bool first_in_row = true;
    for (tt_xy_pair const& worker_slice_shape : worker_slice_shapes) {
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
    std::vector<tt_xy_pair> const& worker_slice_shapes, tt_xy_pair const& tensor_slice_shape) {
    std::vector<tt_xy_pair> worker_slice_offsets;
    worker_slice_offsets.reserve(worker_slice_shapes.size());

    std::uint32_t flattened_idx = 0;

    for (tt_xy_pair const& worker_slice_shape : worker_slice_shapes) {
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
    std::vector<tt_xy_pair> const& worker_slice_shapes, tt_xy_pair const& tensor_slice_shape) {
    return compute_worker_slice_offsets_for_wrapped_tensor_slicer(worker_slice_shapes, tensor_slice_shape);
}

template <class DERIVED_SLICER_T>
std::vector<tt_xy_pair>
RingReduceScatterBaseTensorSlicer<DERIVED_SLICER_T>::create_worker_slice_shapes_for_row_major_layout(
    tt_xy_pair const& tensor_slice_shape_in_elems, uint32_t num_workers, uint32_t max_slice_size_in_elements) {
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
    std::size_t max_width_in_tiles = std::min<std::size_t>(max_slice_size_in_tiles, tensor_slice_shape_in_tiles.x);
    std::size_t max_height_in_tiles = std::min<std::size_t>(max_slice_size_in_tiles, tensor_slice_shape_in_tiles.y);

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
    for (auto const& worker_slice_shape : worker_slice_shapes) {
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
    auto get_padded_worker_slice_size_in_tiles = [](tt_xy_pair const& worker_slice_shape, uint32_t half_cb_n_pages) {
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
        [max_slice_size_in_pages](tt_xy_pair const& worker_slice_shape) {
            TT_ASSERT(worker_slice_shape.x * worker_slice_shape.y <= max_slice_size_in_pages);
        });
    return worker_slice_shapes;
}

std::vector<tt_xy_pair> RingReduceScatterWrappedTensorSlicer::create_worker_slice_shapes_for_tile_layout(
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

    // Assign slices by assuming that the input tensor is flattened into a 1D Shape
    std::size_t optim_worker_slice_len_tiles =
        ceil(total_num_tiles / num_workers);  // Ceil so that the remainder worker will have a smaller slice

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

        auto const& tensor_slice = TensorSlice(
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

        auto const& tensor_slice = TensorSlice(
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
    const Tensor& output_tensor,
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
    std::vector<tt_xy_pair> const& worker_slice_shapes, tt_xy_pair const& tensor_slice_shape) {
    return compute_worker_slice_offsets_for_wrapped_tensor_slicer(worker_slice_shapes, tensor_slice_shape);
}

std::vector<tt_xy_pair> GenericWrappedTensorSlicer::create_worker_slice_shapes_for_tile_layout(
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
    uint32_t total_num_workers)
{
    this->initialize(input_tensor, slice_dim, partition_index, partition_size, total_num_workers);
}

Shape4D<uint32_t> GenericWrappedTensorSlicerV2::calculate_tensor_slice_shape(
    Shape4D<uint32_t> const& input_shape,
    int slice_dim,
    uint32_t partition_size) {

    // Calculate the size of the slice along the given dimension
    uint32_t dim_size = input_shape[slice_dim];
    uint32_t slice_size = dim_size / partition_size;

    // Start with full shape
    Shape4D<uint32_t> slice_shape(input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

    TT_FATAL(slice_dim >= 0 && slice_dim < 4, "Invalid slice dimension. Must be between 0 and 3 but got {}. This should have been normalized to fit within the range", slice_dim);
    slice_shape[slice_dim] = slice_size;

    return slice_shape;
}

Shape4D<uint32_t> GenericWrappedTensorSlicerV2::calculate_tensor_slice_offset(
    Shape4D<uint32_t> const& input_shape,
    int slice_dim,
    uint32_t partition_index) {

    Shape4D<uint32_t> offset(0, 0, 0, 0);

    // Calculate the size of the slice along the given dimension
    uint32_t dim_size = input_shape[slice_dim];
    uint32_t slice_size = dim_size / partition_size;

    TT_FATAL(slice_dim >= 0 && slice_dim < 4, "Invalid slice dimension. Must be between 0 and 3 but got {}. This should have been normalized to fit within the range", slice_dim);
    offset[slice_dim] = partition_index * slice_size;

    return offset;
}

void GenericWrappedTensorSlicerV2::initialize(
    const Tensor& input_tensor,
    int slice_dim,
    uint32_t partition_index,
    uint32_t partition_size,
    uint32_t total_num_workers)
{
    // Configure layout parameters
    this->row_major = (input_tensor.layout() == Layout::ROW_MAJOR);
    this->input_page_size = input_tensor.buffer()->page_size();
    this->partition_index = partition_index;
    this->partition_size = partition_size;

    // Assume everything in Tile layout for now, row major not supported yet
    TT_FATAL(!this->row_major, "Row major not supported yet");

    // Record the input tensor shape
    auto input_shape = input_tensor.padded_shape();
    this->tensor_shape = Shape4D<uint32_t>(input_shape[0], input_shape[1], input_shape[2]/tt::constants::TILE_HEIGHT, input_shape[3]/tt::constants::TILE_WIDTH);

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
        this->tensor_shape, // tensor_shape
        this->tensor_slice_shape, // tensor_slice_shape
        this->tensor_slice_offset, // tensor_slice_offset
        this->worker_slice_shapes[global_worker_index], // worker_slice_shape
        this->worker_slice_offsets[global_worker_index] // worker_slice_offset
    );
}

/* Worker slices and offsets are 4D shapes but flattened to 1D in the last dimension*/

std::vector<Shape4D<uint32_t>> GenericWrappedTensorSlicerV2::compute_worker_slice_offsets(std::vector<Shape4D<uint32_t>> const& worker_slice_shapes) {
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
        Shape4D<uint32_t> const& tensor_slice_shape_in_tiles,
        uint32_t num_workers)
{
    std::vector<Shape4D<uint32_t>> worker_slice_shapes;
    worker_slice_shapes.reserve(num_workers);
    const uint32_t total_num_tiles = tensor_slice_shape_in_tiles.x * tensor_slice_shape_in_tiles.y * tensor_slice_shape_in_tiles.z * tensor_slice_shape_in_tiles.w;
    if (num_workers > total_num_tiles) {
        log_warning(
            tt::LogOp,
            "More workers instantiated than is work to be done. Some workers will be idle and do nothing");
        for (uint32_t w = 0; w < total_num_tiles; ++w) {
            worker_slice_shapes.emplace_back(1,1,1,1);
        }
        for (uint32_t w = total_num_tiles; w < num_workers; ++w) {
            worker_slice_shapes.emplace_back(0,0,0,0);
        }
        return worker_slice_shapes;
    }

    // Assign slices by assuming that the input tensor is flattened into a 1D Shape
    std::size_t optim_worker_slice_len_tiles = std::ceil(static_cast<float>(total_num_tiles) / num_workers); // Ceil so that the remainder worker will have a smaller slice

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
        worker_slice_shapes.back() = Shape4D<uint32_t>(1,1,1,remainder_worker_len_tiles);
    }

    log_trace(tt::LogOp, "--------------------------------");

    return worker_slice_shapes;
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

}  // namespace ccl
}  // namespace ttnn
