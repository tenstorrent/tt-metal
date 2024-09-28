// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"

#include <cstdint>

namespace tt {
namespace tt_metal {

// Forward declarations
class Device;

} // namespace tt_metal
} // namespace tt

namespace ttnn {
namespace ccl {
class WorkerEdmInterfaceArgs;

namespace reduce_scatter_detail {

void emit_ccl_send_slice_sequence_commands(std::vector<TensorSlice> const& slices, std::vector<uint32_t>& args_out);

struct ReduceScatterWorkerArgBuilder {
    ReduceScatterWorkerArgBuilder (
        tt::tt_metal::Device const* device,
        ttnn::ccl::CCLOpConfig const& op_config,
        ttnn::ccl::RingTopology const& topology_config,
        ttnn::ccl::InterleavedTensorWorkerSlice const& worker_input_slice,
        WorkerTransferInfo const& worker_transfer_info,
        ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode,
        std::size_t scatter_dim,
        std::size_t cb_num_pages_per_packet,
        std::optional<uint32_t> receiver_worker_partial_ready_semaphore_id,
        std::size_t num_buffers_per_channel
        );

    std::size_t get_total_num_math_pages(WorkerAttributes const& worker_attrs) const;

    std::vector<uint32_t> generate_reduce_op_kernel_ct_args() const;

    std::vector<uint32_t> generate_reduce_op_kernel_rt_args(WorkerAttributes const& worker_attrs, std::size_t ring_size) const;

    std::vector<uint32_t> generate_receiver_kernel_ct_args() const;

    std::vector<uint32_t> generate_receiver_kernel_rt_args(
        ttnn::ccl::WorkerXY const& edm_core,
        uint32_t edm_core_semaphore_address,
        uint32_t edm_core_buffer_address,
        WorkerAttributes const& worker_attrs) const;

    std::vector<uint32_t> generate_sender_kernel_ct_args() const;

    std::vector<uint32_t> generate_sender_kernel_rt_args(
        WorkerEdmInterfaceArgs const& edm_interface,
        WorkerAttributes const& worker_attrs) const;


    std::vector<uint32_t> generate_line_start_sender_kernel_rt_args(
        WorkerEdmInterfaceArgs const& edm_interface,
        std::size_t scatter_dim,
        WorkerAttributes const& worker_attrs) const;

    std::vector<uint32_t> generate_line_start_sender_kernel_ct_args() const;

    tt::tt_metal::Device const*device;
    ttnn::ccl::RingTopology const topology_config;
    ttnn::ccl::CCLOpConfig const op_config;
    ttnn::ccl::InterleavedTensorWorkerSlice const worker_input_slice;
    WorkerTransferInfo const worker_transfer_info;
    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode;
    uint32_t cb_num_pages_per_packet;
    uint32_t worker_sender_semaphore_id;
    uint32_t worker_receiver_semaphore_id;
    uint32_t num_buffers_per_channel;
    std::optional<uint32_t> receiver_worker_partial_ready_semaphore_id;

    std::size_t scatter_dim;
    bool src_is_dram;
    bool dst_is_dram;
};

} // namespace reduce_scatter_detail
} // namespace ccl
} // namespace ttnn
