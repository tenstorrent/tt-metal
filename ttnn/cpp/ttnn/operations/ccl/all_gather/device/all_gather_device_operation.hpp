// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/device_operation.hpp"
#include "all_gather_device_operation_types.hpp"
#include "all_gather_multicast_factory.hpp"
#include "all_gather_unicast_factory.hpp"
#include "ttnn/operations/ccl/all_gather/shared_with_host/all_gather_walk.hpp"

namespace ttnn::operations::ccl {

struct AllGatherDeviceOperation {
    using operation_attributes_t = AllGatherParams;
    using tensor_args_t = AllGatherInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<AllGatherMulticastFactory, AllGatherUnicastFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

tt::tt_metal::TensorSpec compute_output_specs_helper(
    const Tensor& input_tensor,
    int32_t gather_dim_from_end,
    uint32_t num_devices,
    const std::optional<MemoryConfig>& memory_config);

// The kernels' walk geometry (all_gather_walk.hpp): this device's input pages and the output pages cut into units of
// a quantum that is a multiple of the NOC write alignment and of the read alignment of every source the kernels read,
// or nullopt when the pages cannot be cut so. The unicast relay re-reads its own output, so `unicast` adds the
// output's read alignment.
std::optional<all_gather_walk::WalkGeo> compute_walk_geometry(
    const Tensor& input_tensor,
    const tt::tt_metal::TensorSpec& output_spec,
    int32_t gather_dim_from_end,
    uint32_t num_devices,
    uint32_t packet_size,
    bool unicast);

// The kernels' packet payload cap for this output (all_gather_walk::packet_cap).
uint32_t compute_packet_cap(const all_gather_walk::WalkGeo& geo, const Tensor& output_tensor);

// Reads to keep in flight behind one read barrier: about 30 KB hides DRAM read latency whatever the fabric payload (BH:
// two 15168 B packets per barrier take 24% less worker time per packet than one).
inline constexpr uint32_t default_read_bytes_per_barrier = 30336;

// Packets per CB slot when the walk elements are split over num_slices workers. header_bytes: the fabric packet header
// the writer keeps in the slot, 0 for none (all_gather_walk::slot_offsets).
uint32_t compute_packets_per_slot(
    const all_gather_walk::WalkGeo& geo,
    uint32_t packet_cap,
    uint32_t num_slices,
    uint32_t header_bytes,
    const Tensor& input_tensor,
    uint32_t read_bytes_per_barrier = default_read_bytes_per_barrier);

// Worker cores the op may use: the sub-device's, within sub_core_grid.
CoreRangeSet get_available_worker_cores(const AllGatherParams& args, const Tensor& input_tensor);

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {

Tensor all_gather(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn::prim
