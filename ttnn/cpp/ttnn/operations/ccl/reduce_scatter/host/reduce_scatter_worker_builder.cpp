// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_worker_builder.hpp"
#include <cstdint>
#include <iterator>
#include "hostdevcommon/kernel_structs.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command_host.hpp"

namespace ttnn {
namespace ccl {
namespace reduce_scatter_detail {

ReduceScatterWorkerArgBuilder::ReduceScatterWorkerArgBuilder (
    Device const* device,
    ttnn::ccl::CCLOpConfig const& op_config,
    ttnn::ccl::RingTopology const& topology_config,
    ttnn::ccl::InterleavedTensorWorkerSlice const& worker_input_slice,
    WorkerTransferInfo const& worker_transfer_info,
    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode,
    uint32_t worker_idx,
    uint32_t link,
    uint32_t cb_num_pages_per_packet,
    uint32_t worker_sender_semaphore_id,
    uint32_t worker_receiver_semaphore_id,
    std::optional<uint32_t> receiver_worker_partial_ready_semaphore_id) :
    device(device),
    op_config(op_config),
    topology_config(topology_config),
    worker_input_slice(worker_input_slice),
    worker_transfer_info(worker_transfer_info),
    edm_termination_mode(edm_termination_mode),
    cb_num_pages_per_packet(cb_num_pages_per_packet),
    worker_sender_semaphore_id(worker_sender_semaphore_id),
    worker_receiver_semaphore_id(worker_receiver_semaphore_id) {
    // This algorithm assumes that the worker slices are sized such that they start at the same x offsets for each
    // new row they slice into (as they stride through the tensor)
    std::size_t num_slice_iterations =
        worker_input_slice.compute_num_worker_slice_iterations(worker_transfer_info.num_workers);
    std::size_t worker_slice_num_pages =
        worker_input_slice.worker_slice_shape.x * worker_input_slice.worker_slice_shape.y;
    std::size_t pages_per_full_chunk = worker_transfer_info.get_num_pages_per_full_chunk(link, worker_idx);
    std::size_t num_filler_pages_per_slice = pages_per_full_chunk - (worker_slice_num_pages % pages_per_full_chunk);
    this->total_num_math_pages = (worker_input_slice.get_worker_slice_num_pages() + num_filler_pages_per_slice) *
                                    num_slice_iterations * (topology_config.ring_size - 1);

    log_trace(tt::LogOp, "ReduceScatterWorkerArgBuilder: total_num_math_pages: {}", this->total_num_math_pages);
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_reduce_op_kernel_ct_args() const {
    log_trace(tt::LogOp, "Reduce Scatter Worker CT Args: None");
    return {};
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_reduce_op_kernel_rt_args() const {
    log_trace(tt::LogOp, "generate_reduce_op_kernel_rt_args");

    auto const& args = std::vector<uint32_t>{total_num_math_pages, 1, 0};

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Worker RT Args:");
    log_trace(tt::LogOp, "\tblock_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\ttotal_num_math_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\tacc_to_dst: {}", args.at(i++));

    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_receiver_kernel_ct_args() const {
    auto const& local_input_tensor = this->op_config.get_input_tensor(0);
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);
    auto args = std::vector<uint32_t>{

        static_cast<uint32_t>(this->op_config.is_input_sharded() ? 1 : 0),

        static_cast<uint32_t>(local_input_tensor.memory_config().buffer_type == BufferType::DRAM ? 1 : 0),

        static_cast<uint32_t>(local_input_tensor.memory_config().memory_layout),

        static_cast<uint32_t>(local_output_tensor.memory_config().buffer_type == BufferType::DRAM ? 1 : 0),

        static_cast<uint32_t>(local_output_tensor.memory_config().memory_layout),

        static_cast<uint32_t>(this->topology_config.is_linear ? 1 : 0),
            };

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Receiver Worker CT Args:");
    log_trace(tt::LogOp, "\tis_sharded: {}", args.at(i++));
    log_trace(tt::LogOp, "\tsrc_is_dram: {}", args.at(i++));
    log_trace(tt::LogOp, "\tinput_tensor_memory_layout: {}", args.at(i++));
    log_trace(tt::LogOp, "\tdest_is_dram: {}", args.at(i++));
    log_trace(tt::LogOp, "\toutput_tensor_memory_layout: {}", args.at(i++));
    log_trace(tt::LogOp, "\tis_linear: {}", args.at(i++));

    TT_ASSERT(args.size() == i, "Missed some args");
    if (local_input_tensor.is_sharded()) {
        std::ranges::copy(ShardedAddrGenArgBuilder::emit_ct_args(local_input_tensor), std::back_inserter(args));
        std::ranges::copy(ShardedAddrGenArgBuilder::emit_ct_args(local_output_tensor), std::back_inserter(args));

        ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_input_tensor, "input");
        ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_output_tensor, "output");
    }

    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_receiver_kernel_rt_args(
    ttnn::ccl::WorkerXY const& edm_core,
    uint32_t edm_core_semaphore_address,
    uint32_t edm_core_buffer_address,
    uint32_t link,
    uint32_t worker_index,
    bool is_in_clockwise_direction) const {
    TT_ASSERT(edm_core_semaphore_address > 0);
    TT_ASSERT(edm_core_buffer_address > 0);
    auto const& local_input_tensor = this->op_config.get_input_tensor(0);
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);
    uint32_t starting_ring_index =
        is_in_clockwise_direction ? (this->topology_config.ring_index == 0 ? this->topology_config.ring_size - 1
                                                                            : this->topology_config.ring_index - 1)
                                    : (this->topology_config.ring_index == this->topology_config.ring_size - 1
                                            ? 0
                                            : this->topology_config.ring_index + 1);
    uint32_t num_transfers = this->topology_config.is_linear
                                 ? (this->topology_config.ring_size - (is_in_clockwise_direction ? this->topology_config.ring_index : (this->topology_config.ring_size - 1 - this->topology_config.ring_index)))
                                 : this->topology_config.ring_size;
    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(local_input_tensor.buffer()->address()),
        static_cast<uint32_t>(local_output_tensor.buffer()->address()),
        static_cast<uint32_t>(num_transfers),
        static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)),
        static_cast<uint32_t>(this->op_config.get_page_size()),
        static_cast<uint32_t>(starting_ring_index),
        static_cast<uint32_t>(this->topology_config.ring_size),
        static_cast<uint32_t>(this->worker_receiver_semaphore_id),
        static_cast<uint32_t>(is_in_clockwise_direction ? 1 : 0),
        static_cast<uint32_t>(this->cb_num_pages_per_packet),
        static_cast<uint32_t>(edm_core.x),
        static_cast<uint32_t>(edm_core.y),
        static_cast<uint32_t>(edm_core_semaphore_address),
        static_cast<uint32_t>(edm_core_buffer_address),

        static_cast<uint32_t>(worker_transfer_info.num_workers),

        static_cast<uint32_t>(this->worker_input_slice.tensor_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.tensor_shape.y),

        static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.y),

        static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.y),

        static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.x),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.y),

        this->total_num_math_pages};

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Receiver Worker RT Args:");
    log_trace(tt::LogOp, "\tinput_tensor_address: {}", args.at(i++));
    log_trace(tt::LogOp, "\toutput_tensor_address: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_transfers: {}", args.at(i++));
    log_trace(tt::LogOp, "\tfull_chunk_num_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\tmy_ring_idx: {}", args.at(i++));
    log_trace(tt::LogOp, "\tring_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\tsem_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\tis_clockwise_direction: {}", args.at(i++));
    log_trace(tt::LogOp, "\thalf_cb_n_pages: {}", args.at(i++));

    log_trace(tt::LogOp, "\tedm_core_noc0_core_x: {}", args.at(i++));
    log_trace(tt::LogOp, "\tedm_core_noc0_core_y: {}", args.at(i++));
    log_trace(tt::LogOp, "\tedm_core_semaphore_address: {}", args.at(i++));
    log_trace(tt::LogOp, "\tedm_core_buffer_address: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_concurrent_workers: {}", args.at(i++));

    log_trace(tt::LogOp, "\tinput_tensor_shape.x={}", args.at(i++));
    log_trace(tt::LogOp, "\tinput_tensor_shape.y={}", args.at(i++));
    log_trace(tt::LogOp, "\ttensor_slice_shape.x={}", args.at(i++));
    log_trace(tt::LogOp, "\ttensor_slice_shape.y={}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_shape.x={}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_shape.y={}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_offset.x={}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_offset.y={}", args.at(i++));
    log_trace(tt::LogOp, "\ttotal_num_math_pages={}", args.at(i++));

    TT_ASSERT(args.size() == i, "Missed some args");


    if (local_input_tensor.is_sharded()) {
        std::ranges::copy(ShardedAddrGenArgBuilder::emit_rt_args(device, local_input_tensor), std::back_inserter(args));
        ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_input_tensor, "input");

        if (this->topology_config.is_linear) {
            std::ranges::copy(ShardedAddrGenArgBuilder::emit_rt_args(device, local_output_tensor), std::back_inserter(args));
            ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_output_tensor, "output");
        }
    }

    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_sender_kernel_ct_args() const {
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);

    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(this->op_config.is_input_sharded() ? 1 : 0),
        static_cast<uint32_t>(
            this->op_config.get_output_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0),
        static_cast<uint32_t>(local_output_tensor.memory_config().memory_layout),
        static_cast<uint32_t>(this->topology_config.is_linear)
    };

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Sender Worker CT Args:");
    log_trace(tt::LogOp, "\tis_sharded: {}", args.at(i++));
    log_trace(tt::LogOp, "\tdst_is_dram: {}", args.at(i++));
    log_trace(tt::LogOp, "\ttensor_memory_layout: {}", args.at(i++));
    log_trace(tt::LogOp, "\tis_linear: {}", args.at(i++));
    TT_ASSERT(args.size() == i, "Missed some args");

    if (local_output_tensor.is_sharded()) {
        auto const& shard_ct_args = ShardedAddrGenArgBuilder::emit_ct_args(local_output_tensor);
        std::copy(shard_ct_args.begin(), shard_ct_args.end(), std::back_inserter(args));
    }
    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_sender_kernel_rt_args(
    WorkerEdmInterfaceArgs const& edm_interface,
    uint32_t link,
    uint32_t worker_index,
    std::unordered_map<std::size_t, CoreCoord> const& worker_association_map,
    bool is_clockwise) const {
    TT_ASSERT(edm_interface.edm_semaphore_address > 0);
    TT_ASSERT(edm_interface.edm_buffer_base_address > 0);
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);
    uint32_t num_transfers = this->topology_config.is_linear
        ? (this->topology_config.ring_size - (is_clockwise ? this->topology_config.ring_index : (this->topology_config.ring_size - 1 - this->topology_config.ring_index)))
        : this->topology_config.ring_size - 1;


    bool distance_from_start_of_line = is_clockwise ? this->topology_config.ring_index : this->topology_config.ring_size - 1 - this->topology_config.ring_index;
    bool distance_from_end_of_line = is_clockwise ? this->topology_config.ring_size - 1 - this->topology_config.ring_index : this->topology_config.ring_index;
    bool is_closer_to_start_of_line =
        (distance_from_start_of_line < distance_from_end_of_line) ||
        (distance_from_start_of_line == distance_from_end_of_line && is_clockwise);
    bool signal_reader_on_output_tensor_partial_writes =
            this->topology_config.is_linear && is_closer_to_start_of_line;

    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(local_output_tensor.buffer()->address()),
        static_cast<uint32_t>(edm_interface.edm_buffer_base_address),
        static_cast<uint32_t>(edm_interface.edm_semaphore_address),
        static_cast<uint32_t>(edm_interface.edm_noc_x),
        static_cast<uint32_t>(edm_interface.edm_noc_y),
        static_cast<uint32_t>(this->topology_config.ring_size - 1),  // num_transfers),

        static_cast<uint32_t>(this->op_config.get_page_size()),
        static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)),

        static_cast<uint32_t>(this->worker_sender_semaphore_id),
        static_cast<uint32_t>(this->cb_num_pages_per_packet),

        static_cast<uint32_t>(worker_transfer_info.num_workers),

        // For sender side, all worker slice info is the same except for the tensor shape
        // and for sender side specifically, there is only one tensor_slice_shape for the output
        // tensor (as opposed to `ring_size` tensor_slice_shapes for the input tensor), so we can
        // directly use it as the output tensor shape
        static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.y),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.y),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.x),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.y),

        total_num_math_pages,
        static_cast<uint32_t>(signal_reader_on_output_tensor_partial_writes ? 1 : 0)};

    if (signal_reader_on_output_tensor_partial_writes) {
        TT_ASSERT(receiver_worker_partial_ready_semaphore_id.has_value());
        auto associated_worker_core = worker_association_map.at(link * this->worker_transfer_info.num_workers + worker_index);
        auto const& worker_core_xy = this->device->worker_core_from_logical_core(associated_worker_core);

        args.push_back(static_cast<uint32_t>(worker_core_xy.x));
        args.push_back(static_cast<uint32_t>(worker_core_xy.y));
        args.push_back(static_cast<uint32_t>(receiver_worker_partial_ready_semaphore_id.value()));
    }

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Sender Worker RT Args (signal_reader_on_output_tensor_partial_writes={}):", signal_reader_on_output_tensor_partial_writes);
    log_trace(tt::LogOp, "\tdst_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\teth_sender_l1_base_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\teth_sender_l1_sem_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\teth_sender_noc_x: {}", args.at(i++));
    log_trace(tt::LogOp, "\teth_sender_noc_y: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_transfers: {}", args.at(i++));
    log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\tfull_chunk_num_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\twriter_send_sem_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\thalf_cb_n_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_concurrent_workers: {}", args.at(i++));

    log_trace(tt::LogOp, "\toutput_tensor_shape.x: {}", args.at(i++));
    log_trace(tt::LogOp, "\toutput_tensor_shape.y: {}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_shape.x: {}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_shape.y: {}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_offset.x: {}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_offset.y: {}", args.at(i++));

    log_trace(tt::LogOp, "\ttotal_num_math_pages={}", args.at(i++));
    log_trace(tt::LogOp, "\tsignal_reader_on_output_tensor_partial_writes={}", args.at(i++));

    if (signal_reader_on_output_tensor_partial_writes) {
        log_trace(tt::LogOp, "\teth_receiver_noc_x: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_receiver_noc_y: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_receiver_sem_addr: {}", args.at(i++));
    }

    TT_ASSERT(args.size() == i, "Missed some args");

    if (local_output_tensor.is_sharded()) {
        auto const& shard_rt_args = ShardedAddrGenArgBuilder::emit_rt_args(device, local_output_tensor);
        std::copy(shard_rt_args.begin(), shard_rt_args.end(), std::back_inserter(args));

        ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_output_tensor, "output");
    }
    return args;
}


std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_line_start_sender_kernel_rt_args(
    WorkerEdmInterfaceArgs const& edm_interface,
    std::size_t scatter_dim,
    std::size_t link,
    std::size_t worker_index) const
{
    static constexpr std::size_t max_args = 1024; // need to get from tt_metal

    std::size_t num_commands = this->topology_config.ring_size - 1;
    auto &input_tensor = this->op_config.get_input_tensor(0);
    TT_ASSERT(input_tensor.get_legacy_shape().size() == 4, "Only 4D tensors are supported for reduce scatter");
    ttnn::ccl::Shape4D<uint32_t> input_tensor_shape = {input_tensor.get_legacy_shape()[0], input_tensor.get_legacy_shape()[1],input_tensor.get_legacy_shape()[2],input_tensor.get_legacy_shape()[3]};

    std::vector<uint32_t> args = {
        static_cast<uint32_t>(input_tensor.buffer()->address()),
        static_cast<uint32_t>(num_commands)
    };
    std::ranges::copy(ttnn::ccl::emit_runtime_args(input_tensor_shape), std::back_inserter(args));
    // ttnn::ccl::log_runtime_args(input_tensor_shape, "tensor_shape");

    std::ranges::copy(ttnn::ccl::emit_runtime_args(input_tensor_shape), std::back_inserter(args)); // window shape
    // ttnn::ccl::log_runtime_args(input_tensor_shape, "tensor_slice_shape");

    std::ranges::copy(ttnn::ccl::emit_runtime_args(edm_interface), std::back_inserter(args));
    // ttnn::ccl::log_runtime_args(edm_interface, "edm_interface");

    std::ranges::copy(std::vector<uint32_t>{this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)}, std::back_inserter(args));
    std::ranges::copy(std::vector<uint32_t>{this->op_config.get_page_size()}, std::back_inserter(args));
    std::ranges::copy(ttnn::ccl::emit_address_generator_runtime_args(this->device, input_tensor), std::back_inserter(args));
    std::ranges::copy(std::vector<uint32_t>{this->worker_sender_semaphore_id}, std::back_inserter(args));

    // If we are on device zero, we send n-1 chunks in ascending order

    auto const& tensor_shape = this->worker_input_slice.tensor_shape;
    auto const& tensor_slice_shape = this->worker_input_slice.tensor_slice_shape;

    auto num_slices = topology_config.ring_size - 1;
    auto start_slice_index = topology_config.ring_size == 0 ? 0 : topology_config.ring_index - 1;
    TT_ASSERT(start_slice_index == 0 || start_slice_index == topology_config.ring_size - 1);
    auto end_slice_index_exclusive = topology_config.ring_size - start_slice_index; //-  start_slice_index == 0 ? num_slices : 0;

    // Add the command args
    auto const& slices = generate_slice_sequence_on_dim(
        tensor_shape,
        worker_input_slice.worker_slice_shape,
        scatter_dim,
        num_slices,
        start_slice_index,
        end_slice_index_exclusive
    );
    for (auto const& slice : slices) {
        ttnn::ccl::cmd::CclCommand c = {
            Shape4D<uint32_t>{1, 1, slice.tensor_slice_shape.y, slice.tensor_slice_shape.x},
            Shape4D<uint32_t>{1, 1, slice.tensor_slice_offset.y, slice.tensor_slice_offset.x},
            Shape4D<uint32_t>{1, 1, slice.worker_slice_offset.y, slice.worker_slice_offset.x},
            this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)
        };
        std::ranges::copy(add_ccl_command_to_args(c), std::back_inserter(args));
    }

    TT_FATAL(args.size() < max_args, "Too many command args. The cluster size is too large for reduce scatter");

    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_line_start_sender_kernel_ct_args() const
{
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).memory_config().memory_layout), // tensor memory layout
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).buffer()->buffer_type()), // buffer type
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).layout()), // page layout
        static_cast<uint32_t>(this->edm_termination_mode), // (EDM) termination mode
        static_cast<uint32_t>(tt::CB::c_in0) // cb_id
    };

    return args;
}

} // namespace reduce_scatter_detail
} // namespace ccl
} // namespace ttnn
