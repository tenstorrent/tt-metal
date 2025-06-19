// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/distributed/types.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

namespace ttnn {

enum AllGatherBidirectionalMode {
    // Splits the tensor into two and sends each half in opposite directions
    // the full width around the ring
    SPLIT_TENSOR,
    // Doesn't split the tensor and sends the full tensor in both directions,
    // half-way around the ring
    FULL_TENSOR
};

using ccl::EriscDatamoverBuilder;

class AllGatherConfig {
    static AllGatherBidirectionalMode choose_bidirectional_mode(Tensor const& input_tensor, bool fuse_op);

   public:
       AllGatherConfig(
           const Tensor& input_tensor,
           const Tensor& output_tensor,
           uint32_t dim,
           uint32_t ring_size,
           uint32_t num_links,
           ccl::Topology topology,
           std::size_t num_buffers_per_worker,
           bool fuse_op = false,
           std::optional<size_t> user_defined_num_workers = std::nullopt);

       uint32_t get_erisc_handshake_address() const { return this->erisc_handshake_address; }

       uint32_t get_num_eth_buffers_per_edm() const { return this->num_eth_buffers; }
       uint32_t get_num_workers_per_link() const { return this->num_workers_per_link; }
       uint32_t get_num_workers() const { return this->num_workers_per_link * this->num_links; }

       uint32_t get_eth_buffer_size() const { return this->eth_buffer_size; }

       uint32_t get_eth_sems_l1_base_byte_address() const { return this->eth_sems_l1_base_byte_address; }

       uint32_t get_eth_buffers_l1_base_byte_address() const { return this->eth_buffers_l1_base_byte_address; }

       uint32_t get_semaphore_size() const { return this->semaphore_size; }
       std::size_t get_num_buffers_per_channel() const { return this->num_edm_buffers_per_channel; }

       uint32_t get_num_edm_channels_in_clockwise_direction() const {
           return this->enable_bidirectional ? this->num_workers_per_link / 2 : this->num_workers_per_link;
    }
    uint32_t get_ring_size() const { return this->ring_size; }
    bool is_payload_and_channel_sync_merged() const { return enable_merged_payload_and_channel_sync;}
    bool is_buffer_in_clockwise_ring(const uint32_t buffer_index) const {
        // For now we split it as lower half => clockwise, upper half => counter-clockwise
        // This is slightly suboptimal since the non-full-chunks go to the upper half.
        // A more optimal split would be round robin
        return this->enable_bidirectional ?
            buffer_index < get_num_edm_channels_in_clockwise_direction() :
            true;
    }
    AllGatherBidirectionalMode get_bidirectional_mode() const { return this->bidirectional_mode; }
    uint32_t get_num_edm_channels_in_counter_clockwise_direction() const {
        // return all_gather_buffer_params::enable_bidirectional ? all_gather_buffer_params::num_buffers - all_gather_buffer_params::num_buffers / 2 : 0;
        // Force all through counter-clockwise direction
        return this->num_workers_per_link - this->get_num_edm_channels_in_clockwise_direction();
    }

    bool is_input_dram() const { return input_is_dram; }
    bool is_output_dram() const { return output_is_dram; }


    void print() const {
        log_trace(tt::LogOp, "AllGatherConfig: (");
        log_trace(tt::LogOp, "\tis_sharded: {}", is_sharded);
        log_trace(tt::LogOp, "\terisc_handshake_address: {}", erisc_handshake_address);
        log_trace(tt::LogOp, "\tnum_buffers: {}", num_eth_buffers);
        log_trace(tt::LogOp, "\tnum_workers_per_link: {}", num_workers_per_link);
        log_trace(tt::LogOp, "\tnum_edm_buffers_per_channel: {}", num_edm_buffers_per_channel);
        log_trace(tt::LogOp, "\teth_buffer_size: {}", eth_buffer_size);
        log_trace(tt::LogOp, "\tsemaphore_size: {}", semaphore_size);
        log_trace(tt::LogOp, "\tsemaphore_offset: {}", semaphore_offset);
        log_trace(tt::LogOp, "\teth_buffers_l1_base_byte_address: {}", eth_buffers_l1_base_byte_address);
        log_trace(tt::LogOp, "\teth_sems_l1_base_byte_address: {}", eth_sems_l1_base_byte_address);
        log_trace(tt::LogOp, "\tenable_bidirectional: {}", enable_bidirectional);
        log_trace(tt::LogOp, ")");
    }

   private:
    const uint32_t erisc_handshake_address;
    uint32_t ring_size;
    uint32_t num_links;
    uint32_t num_eth_buffers;
    uint32_t num_workers_per_link;
    uint32_t num_edm_buffers_per_channel;
    uint32_t eth_buffer_size;
    uint32_t semaphore_size;
    uint32_t semaphore_offset;
    uint32_t eth_buffers_l1_base_byte_address;
    uint32_t eth_sems_l1_base_byte_address;
    const ccl::Topology topology;
    AllGatherBidirectionalMode bidirectional_mode;
    bool is_sharded;
    bool enable_bidirectional;
    const bool input_is_dram;
    const bool output_is_dram;
    const bool enable_merged_payload_and_channel_sync;
};

struct AllGather {
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const std::optional<size_t> user_defined_num_workers;
    const std::optional<size_t> user_defined_num_buffers_per_channel;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    std::optional<uint32_t> cluster_axis;
    std::vector<IDevice*> devices;
    const distributed::MeshDevice* mesh_device = nullptr;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& mesh_coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

namespace ccl{
namespace all_gather_detail{
AllGather create_all_gather_struct(
    const Tensor& input_tensor,
    uint32_t dim,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<size_t> user_defined_num_workersm,
    std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    ccl::Topology topology);
} // namespace all_gather_detail
} // namespace ccl

// All Gather Variants
tt::tt_metal::operation::ProgramWithCallbacks all_gather_full_shard_grid(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    std::optional<size_t> user_defined_num_workers,
    std::optional<size_t> user_defined_num_buffers_per_channel,
    std::optional<chip_id_t> receiver_device_id,
    std::optional<chip_id_t> sender_device_id,
    ccl::Topology topology);
tt::tt_metal::operation::ProgramWithCallbacks all_gather_multi_core_with_workers(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    chip_id_t target_device_id,
    std::optional<chip_id_t> receiver_device_id,
    std::optional<chip_id_t> sender_device_id,
    ccl::Topology topology,
    std::optional<size_t> user_defined_num_workers,
    std::optional<size_t> user_defined_num_buffers_per_channel);
tt::tt_metal::operation::ProgramWithCallbacks all_gather_multi_core_with_workers_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    chip_id_t target_device_id,
    std::optional<chip_id_t> receiver_device_id,
    std::optional<chip_id_t> sender_device_id,
    ccl::Topology topology,
    std::optional<size_t> user_defined_num_workers,
    std::optional<size_t> user_defined_num_buffers_per_channel,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    CoreCoord core_grid_offset = CoreCoord(0, 0));

namespace operations {
namespace ccl {

Tensor all_gather(
    const Tensor& input_tensor,
    int32_t dim,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> user_defined_num_workers = std::nullopt,
    std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);

std::vector<Tensor> all_gather(
    const std::vector<Tensor>& input_tensors,
    int32_t dim,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> user_defined_num_workers = std::nullopt,
    std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);

Tensor all_gather(
    const Tensor& input_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> user_defined_num_workers = std::nullopt,
    std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear);

std::vector<Tensor> all_gather(
    const std::vector<Tensor>& input_tensors,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> user_defined_num_workers = std::nullopt,
    std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear);

} // namespace ccl
} // namespace operations

}  // namespace ttnn
