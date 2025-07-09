// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn {

AllGatherBidirectionalMode AllGatherConfig::choose_bidirectional_mode(const Tensor& input_tensor, bool fuse_op) {
    if (fuse_op) {
        return AllGatherBidirectionalMode::FULL_TENSOR;
    }

    std::size_t eth_l1_capacity = tt::tt_metal::hal::get_erisc_l1_unreserved_size();
    std::size_t tensor_size_bytes = input_tensor.physical_volume() * input_tensor.element_size();
    // This is currently a guestimate. We need a lot more hard data to identify where this dividing line is.
    bool perf_degradation_from_full_tensor_mode = tensor_size_bytes > (2 * eth_l1_capacity);
    if (perf_degradation_from_full_tensor_mode) {
        return AllGatherBidirectionalMode::SPLIT_TENSOR;
    }
    return AllGatherBidirectionalMode::FULL_TENSOR;
}

AllGatherConfig::AllGatherConfig(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    uint32_t dim,
    uint32_t ring_size,
    uint32_t num_links,
    ttnn::ccl::Topology topology,
    std::size_t num_edm_buffers_per_channel,
    bool fuse_op,
    const std::optional<size_t> user_defined_num_workers) :
    num_links(num_links),
    semaphore_size(32),
    ring_size(ring_size),

    erisc_handshake_address(tt::round_up(tt::tt_metal::hal::get_erisc_l1_unreserved_base(), 16)),
    topology(topology),
    enable_bidirectional(topology == ttnn::ccl::Topology::Ring),

    input_is_dram(input_tensor.buffer()->buffer_type() == BufferType::DRAM),
    output_is_dram(output_tensor.buffer()->buffer_type() == BufferType::DRAM),

    bidirectional_mode(choose_bidirectional_mode(input_tensor, fuse_op)),
    enable_merged_payload_and_channel_sync(true),
    num_edm_buffers_per_channel(num_edm_buffers_per_channel) {
    TT_FATAL(num_edm_buffers_per_channel > 0, "num_edm_buffers_per_channel must be > 0");
    TT_ASSERT(erisc_handshake_address >= tt::tt_metal::hal::get_erisc_l1_unreserved_base());
    TT_ASSERT(erisc_handshake_address < tt::tt_metal::hal::get_erisc_l1_unreserved_base() + 16);
    TT_ASSERT((erisc_handshake_address & (16 - 1)) == 0);
    if (input_tensor.layout() == Layout::TILE && dim != 3) {
        // See issue #6448
        int outer_dims_size = 1;
        for (std::size_t i = 0; i < dim; i++) {
            outer_dims_size *= input_tensor.padded_shape()[i];
        }
        if (outer_dims_size > 1) {
            this->enable_bidirectional = false;
        }
    }

    // "duplicate" directions are a short hand to enable linear/mesh all-gather topologies with
    // less code-changes. Ideally a new concept is added amongst "num_eth_buffers", "num_workers_per_link", etc.
    uint32_t num_duplicate_directions =
        (topology == ttnn::ccl::Topology::Ring && bidirectional_mode != AllGatherBidirectionalMode::FULL_TENSOR) ? 1
                                                                                                                 : 2;

    uint32_t total_l1_buffer_space = tt::tt_metal::hal::get_erisc_l1_unreserved_size();

    this->is_sharded = input_tensor.is_sharded();
    if (user_defined_num_workers.has_value()) {
        this->num_eth_buffers = user_defined_num_workers.value() / num_duplicate_directions;
    } else {
        this->num_eth_buffers =
            (this->enable_bidirectional ? 8 /*1*/ : (topology != ttnn::ccl::Topology::Linear ? 8 : 4));
    }

    constexpr std::int32_t MAX_NUM_CONCURRENT_TRANSACTIONS = 8;
    if (bidirectional_mode == AllGatherBidirectionalMode::FULL_TENSOR) {
        this->num_eth_buffers =
            std::min(this->num_eth_buffers, MAX_NUM_CONCURRENT_TRANSACTIONS / num_duplicate_directions);
    }

    this->num_workers_per_link = this->num_eth_buffers;
    this->eth_sems_l1_base_byte_address = this->erisc_handshake_address + 16 * 3;  // 16;
    // Really should be called offset_after_semaphore_region
    this->semaphore_offset =
        this->semaphore_size * this->num_eth_buffers *
        num_duplicate_directions;  // TODO: Remove this once dedicated semaphore space for user kernels are added
    this->eth_buffers_l1_base_byte_address = this->eth_sems_l1_base_byte_address + this->semaphore_offset;

    std::size_t channel_sync_bytes_overhead = (enable_merged_payload_and_channel_sync * 16);
    uint32_t const page_size = input_tensor.buffer()->page_size();
    std::size_t l1_per_buffer_region =
        ((total_l1_buffer_space - this->semaphore_offset) /
         (this->num_eth_buffers * num_duplicate_directions * this->num_edm_buffers_per_channel)) -
        channel_sync_bytes_overhead;
    this->eth_buffer_size = tt::round_down(l1_per_buffer_region, page_size);

    TT_FATAL(
        (this->eth_buffer_size + channel_sync_bytes_overhead) *
                    (this->num_eth_buffers * num_duplicate_directions * this->num_edm_buffers_per_channel) +
                this->semaphore_offset <=
            total_l1_buffer_space,
        "Error");
    TT_FATAL(
        eth_buffer_size == 0 or (this->num_eth_buffers * num_duplicate_directions) <= MAX_NUM_CONCURRENT_TRANSACTIONS,
        "Error");
}

void AllGather::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    // TODO: This can be removed by passing two page sizes, actual and aligned to be used for address offsets
    // Buffer sizes also need to take this aligned page size into consideration
    // TODO: Validate ring
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());

    // Sharding Config checks
    bool input_sharded = input_tensor.is_sharded();
    if (input_sharded) {
        // TODO(snijjar)
    }
}

std::vector<ttnn::TensorSpec> AllGather::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = input_tensors[0].logical_shape();
    output_shape[this->dim] *= this->ring_size;

    const auto& input_tensor = input_tensors[0];
    TensorSpec spec(
        output_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config));
    if (this->output_mem_config.is_sharded()) {
        return {TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config))};
    }
    return std::vector<TensorSpec>(input_tensors.size(), spec);
}

std::vector<Tensor> AllGather::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, {});
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllGather::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllGather::create_program_at(
    const ttnn::MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto target_device = input_tensors.at(0).mesh_device() ? input_tensors.at(0).mesh_device()->get_device(mesh_coord)
                                                           : input_tensors.at(0).device();
    ccl::SenderRecieverConfig config =
        this->cluster_axis.has_value()
            ? ccl::get_device_sender_receiver_config_in_ring(mesh_coord, mesh_device, *cluster_axis, ring_size)
            : ccl::get_device_sender_receiver_config(target_device, this->devices, topology);

    return all_gather_multi_core_with_workers(
        input_tensors[0],
        output_tensors[0],
        this->dim,
        this->num_links,
        this->ring_size,
        config.device_index,
        target_device->id(),
        config.receiver_device_id,
        config.sender_device_id,
        this->topology,
        this->user_defined_num_workers,
        this->user_defined_num_buffers_per_channel);
}

namespace operations {
namespace ccl {

namespace {
Tensor all_gather_impl(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology,
    const std::vector<IDevice*>& devices) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "all_gather op is only supported for Fast Dispatch");
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "all_gather op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    int32_t rank = input_tensor.logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    ttnn::SmallVector<uint32_t> unpad_elements = {
        input_tensor.logical_shape()[-4],
        input_tensor.logical_shape()[-3],
        input_tensor.logical_shape()[-2],
        input_tensor.logical_shape()[-1]};

    const uint32_t unpadded_w = input_tensor.logical_shape()[-1];
    const uint32_t unpadded_h = input_tensor.logical_shape()[-2];

    const uint32_t w_pad = tt::round_up(unpadded_w, tt::constants::TILE_WIDTH) - unpadded_w;
    const uint32_t h_pad = tt::round_up(unpadded_h, tt::constants::TILE_HEIGHT) - unpadded_h;
    bool needs_padding = input_tensor.layout() == Layout::TILE && (h_pad != 0 || w_pad != 0);

    Tensor input_tensor_padded = input_tensor;
    if (needs_padding) {
        ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padding = {{0, 0}, {0, 0}, {0, h_pad}, {0, w_pad}};
        DataType original_dtype = input_tensor.dtype();
        if (input_tensor.dtype() != DataType::BFLOAT16 && input_tensor.dtype() != DataType::FLOAT32) {
            input_tensor_padded = ttnn::typecast(input_tensor_padded, DataType::BFLOAT16);
        }
        input_tensor_padded = ttnn::pad(input_tensor_padded, padding, 0, false, std::nullopt);

        if (original_dtype != input_tensor.dtype()) {
            input_tensor_padded = ttnn::typecast(input_tensor_padded, original_dtype);
        }
    }

    auto output_tensor = tt::tt_metal::operation::run(
        ttnn::AllGather{
            .dim = gather_dim,
            .num_links = num_links,
            .ring_size = num_devices,
            .user_defined_num_workers = user_defined_num_workers,
            .user_defined_num_buffers_per_channel = user_defined_num_buffers_per_channel,
            .output_mem_config = memory_config.value_or(input_tensor_padded.memory_config()),
            .topology = ccl_topology,
            .cluster_axis = std::nullopt,
            .devices = devices},
        {input_tensor_padded});

    if (needs_padding) {
        return ttnn::ccl::unpad_output_tensor(output_tensor, num_devices, unpad_elements, dim).at(0);
    } else {
        return output_tensor.at(0);
    }
}
Tensor all_gather_impl(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    TT_FATAL(
        topology == ttnn::ccl::Topology::Linear,
        "This all_gather API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    return tt::tt_metal::operation::run(
               ttnn::AllGather{
                   .dim = gather_dim,
                   .num_links = num_links,
                   .ring_size = num_devices,
                   .user_defined_num_workers = user_defined_num_workers,
                   .user_defined_num_buffers_per_channel = user_defined_num_buffers_per_channel,
                   .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                   .topology = topology,
                   .cluster_axis = cluster_axis,
                   .devices = {},
                   .mesh_device = &mesh_device},
               {input_tensor})
        .at(0);
}
}  // namespace

Tensor all_gather(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    return all_gather_impl(
        input_tensor,
        dim,
        num_links,
        memory_config,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        topology,
        ttnn::ccl::get_active_physical_devices(input_tensor));
}

std::vector<Tensor> all_gather(
    const std::vector<Tensor>& input_tensors,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    std::vector<IDevice*> devices;
    devices.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        devices.push_back(input_tensor.device());
    }

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        output_tensors.push_back(all_gather_impl(
            input_tensor,
            dim,
            num_links,
            memory_config,
            user_defined_num_workers,
            user_defined_num_buffers_per_channel,
            topology,
            devices));
    }
    return output_tensors;
}

Tensor all_gather(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    return all_gather_impl(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        num_links,
        memory_config,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        topology);
}

std::vector<Tensor> all_gather(
    const std::vector<Tensor>& input_tensors,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        output_tensors.push_back(all_gather_impl(
            input_tensor,
            dim,
            cluster_axis,
            mesh_device,
            num_links,
            memory_config,
            user_defined_num_workers,
            user_defined_num_buffers_per_channel,
            topology));
    }
    return output_tensors;
}

}  // namespace ccl
}  // namespace operations

}  // namespace ttnn
