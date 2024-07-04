// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "tensor/tensor.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

namespace ttnn {

enum AllGatherMode {
    RING_INTERLEAVED,
    FULL_WORKER_GRID_SHARDED,
    SINGLE_TILE_HIGH_WIDTH_SHARDED
};

namespace all_gather_op {
using tt::tt_metal::ccl::Topology;
}; // namespace all_gather_op

using tt::tt_metal::ccl::EriscDatamoverBuilder;

AllGatherMode choose_all_gather_mode(Tensor const& input_tensor, Tensor const& output_tensor, uint32_t dim);

class AllGatherConfig {
   public:
    AllGatherConfig(Tensor const& input_tensor, Tensor const& output_tensor, uint32_t dim, uint32_t ring_size, uint32_t num_links, all_gather_op::Topology topology) :
        num_links(num_links),
        semaphore_size(32),
        ring_size(ring_size),

        erisc_handshake_address(tt::round_up(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, 16)),
        topology(topology),
        enable_bidirectional(topology == all_gather_op::Topology::Ring),

        input_is_dram(input_tensor.buffer()->buffer_type() == BufferType::DRAM),
        output_is_dram(output_tensor.buffer()->buffer_type() == BufferType::DRAM),

        mode(choose_all_gather_mode(input_tensor, output_tensor, dim))
    {
        TT_ASSERT(erisc_handshake_address >= eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);
        TT_ASSERT(erisc_handshake_address < eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16);
        TT_ASSERT((erisc_handshake_address & (16-1)) == 0);
        if (input_tensor.get_layout() == Layout::TILE && dim != 3) {
            // See issue #6448
            int outer_dims_size = 1;
            for (std::size_t i = 0; i < dim; i++) {
                outer_dims_size *= input_tensor.get_legacy_shape()[i];
            }
            if (outer_dims_size > 1) {
                this->enable_bidirectional = false;
            }
        }

        // "duplicate" directions are a short hand to enable linear/mesh all-gather topologies with
        // less code-changes. Ideally a new concept is added amongst "num_eth_buffers", "num_workers_per_link", etc.
        uint32_t num_duplicate_directions = topology == all_gather_op::Topology::Ring ? 1 : 2;

        constexpr uint32_t total_l1_buffer_space = eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

        this->is_sharded = input_tensor.is_sharded();
        this->num_eth_buffers = (this->enable_bidirectional ? 8 : (this->is_sharded && topology != all_gather_op::Topology::Linear ? 8 : 4));
        if (this->is_sharded) {
            this->num_eth_buffers = std::min(this->num_eth_buffers, input_tensor.shard_spec()->num_cores());
            if ((input_tensor.shard_spec()->num_cores() / this->num_eth_buffers) % (ring_size) != 0 &&
                (ring_size % (input_tensor.shard_spec()->num_cores() / this->num_eth_buffers) != 0)) {
                // Currently don't support misalignment here
                this->num_eth_buffers = 1;
            }
            log_trace(tt::LogOp, "this->num_buffers: {}", this->num_eth_buffers);
        }

        this->num_workers_per_link = this->num_eth_buffers;
        this->eth_sems_l1_base_byte_address = this->erisc_handshake_address + 16 * 3;//16;
        this->semaphore_offset = this->semaphore_size * this->num_eth_buffers * num_duplicate_directions; // TODO: Remove this once dedicated semaphore space for user kernels are added
        this->eth_buffers_l1_base_byte_address = this->eth_sems_l1_base_byte_address + this->semaphore_offset;

        uint32_t const page_size = input_tensor.buffer()->page_size();
        this->eth_buffer_size = tt::round_down((total_l1_buffer_space - this->semaphore_offset) / (this->num_eth_buffers * num_duplicate_directions), page_size);

        TT_FATAL(eth_buffer_size == 0 or (this->num_eth_buffers * num_duplicate_directions) <= eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS);
        TT_FATAL(this->eth_buffer_size * (this->num_eth_buffers * num_duplicate_directions) + this->semaphore_offset <= total_l1_buffer_space);

        // FIXME: dynamically select the number and size of each buffer based on tensor attributes, link count, ring size, etc.
        // Erisc is able to keep up with workers up to around 17-20 GBps bidirectional (numbers still being locked down)
        // depending on payload size. In general, the smaller each eth buffer, the more overhead per send, and the larger each
        // buffer, the less overhead. However, larger buffers are less desirable for smaller tensors as the first send is delayed
        // until a larger percentage of the overall tensor has landed in the erisc buffer (which can impact latency negatively)
        // and for smaller tensors, latency can be more dominant than throughput with respect to end-to-end runtime.
        // Additionally, tensor layout and location can affect worker throughput. Based on loose empirical testing,
        // if workers are in RowMajor or DRAM tile layout (maybe L1 too - need to measure), it's preffered add more workers
        // (8) with smaller buffers so each worker can keep up with erisc.
    }

    uint32_t get_erisc_handshake_address() const { return this->erisc_handshake_address; }

    uint32_t get_semaphores_offset() const { return this->semaphore_offset; }
    uint32_t get_num_eth_buffers_per_edm() const { return this->num_eth_buffers; }
    uint32_t get_num_workers_per_link() const { return this->num_workers_per_link; }
    uint32_t get_num_workers() const { return this->num_workers_per_link * this->num_links; }

    uint32_t get_eth_buffer_size() const { return this->eth_buffer_size; }

    uint32_t get_eth_sems_l1_base_byte_address() const { return this->eth_sems_l1_base_byte_address; }

    uint32_t get_eth_buffers_l1_base_byte_address() const { return this->eth_buffers_l1_base_byte_address; }

    uint32_t get_semaphore_size() const { return this->semaphore_size; }

    uint32_t get_num_edm_channels_in_clockwise_direction() const {
        return this->enable_bidirectional ?
            this->num_workers_per_link / 2 :
            this->num_workers_per_link;
    }
    uint32_t get_ring_size() const { return this->ring_size; }
    bool is_buffer_in_clockwise_ring(const uint32_t buffer_index) const {
        // For now we split it as lower half => clockwise, upper half => counter-clockwise
        // This is slightly suboptimal since the non-full-chunks go to the upper half.
        // A more optimal split would be round robin
        return this->enable_bidirectional ?
            buffer_index < get_num_edm_channels_in_clockwise_direction() :
            true;
    }
    uint32_t get_num_edm_channels_in_counter_clockwise_direction() const {
        // return all_gather_buffer_params::enable_bidirectional ? all_gather_buffer_params::num_buffers - all_gather_buffer_params::num_buffers / 2 : 0;
        // Force all through counter-clockwise direction
        return this->num_workers_per_link - this->get_num_edm_channels_in_clockwise_direction();
    }

    bool is_input_dram() const { return input_is_dram; }
    bool is_output_dram() const { return output_is_dram; }

    AllGatherMode get_mode() const { return mode; }

    void print() const {
        log_trace(tt::LogOp, "AllGatherConfig: (");
        log_trace(tt::LogOp, "\tis_sharded: {}", is_sharded);
        log_trace(tt::LogOp, "\terisc_handshake_address: {}", erisc_handshake_address);
        log_trace(tt::LogOp, "\tnum_buffers: {}", num_eth_buffers);
        log_trace(tt::LogOp, "\tnum_workers_per_link: {}", num_workers_per_link);
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
    uint32_t eth_buffer_size;
    uint32_t semaphore_size;
    uint32_t semaphore_offset;
    uint32_t eth_buffers_l1_base_byte_address;
    uint32_t eth_sems_l1_base_byte_address;
    const all_gather_op::Topology topology;
    AllGatherMode mode;
    bool is_sharded;
    bool enable_bidirectional;
    const bool input_is_dram;
    const bool output_is_dram;
};

struct RingInterleavedAllGatherVariantConfig : public AllGatherConfig {

    std::string const& send_reader_kernel_path = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_interleaved_ring_gather_send_reader.cpp";
    std::string const& sender_writer_kernel_path = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_interleaved_ring_gather_send_writer.cpp";
    std::string const& receiver_reader_kernel_path = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_interleaved_ring_gather_receive_reader.cpp";
    std::string const& receiver_writer_kernel_path = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_interleaved_ring_gather_receive_writer.cpp";
};

struct SingleTileHighWidthShardedAllGatherVariantConfig : public AllGatherConfig {

    std::string const& send_reader_kernel_path = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_sharded_ring_gather_send_reader.cpp";
    std::string const& sender_writer_kernel_path = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_sharded_ring_gather_send_writer.cpp";
    std::string const& receiver_reader_kernel_path = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_sharded_ring_gather_receive_reader.cpp";
    std::string const& receiver_writer_kernel_path = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_sharded_ring_gather_receive_writer.cpp";
};

struct FullWorkerGridShardedAllGatherVariantConfig : public AllGatherConfig {

    std::string const& reader_kernel = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_sharded_all_shard_workers_ring_gather_reader.cpp";
    std::string const& writer_kernel = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_sharded_all_shard_workers_ring_gather_writer.cpp";
};

struct AllGather {
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<chip_id_t> receiver_device_id;
    const std::optional<chip_id_t> sender_device_id;
    const MemoryConfig output_mem_config;
    const all_gather_op::Topology topology;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
};

// All Gather Variants
operation::ProgramWithCallbacks all_gather_full_shard_grid(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    all_gather_op::Topology topology);
operation::ProgramWithCallbacks all_gather_multi_core_with_workers(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    all_gather_op::Topology topology);

struct ShardedAllGatherConfig {
    ShardedAllGatherConfig(Tensor const& input_tensor, Tensor const& output_tensor, uint32_t dim)
        {
        this->is_sharding_enabled = input_tensor.is_sharded();
        if (!this->is_sharding_enabled) {
            return;
        }

        TT_ASSERT(input_tensor.get_legacy_shape()[2] / TILE_HEIGHT > 0);
        TT_ASSERT(input_tensor.get_legacy_shape()[3] / TILE_WIDTH > 0);
        TT_ASSERT(output_tensor.get_legacy_shape().rank() == 4, "Assumes rank 4");
        TT_ASSERT(input_tensor.get_legacy_shape().rank() == 4, "Assumes rank 4");

        switch(input_tensor.memory_config().memory_layout) {
            case TensorMemoryLayout::WIDTH_SHARDED:
                this->shard_type = tt::tt_metal::ccl::ShardType::Width;
                break;
            case TensorMemoryLayout::BLOCK_SHARDED:
                this->shard_type = tt::tt_metal::ccl::ShardType::Block;
                break;
            case TensorMemoryLayout::HEIGHT_SHARDED:
                this->shard_type = tt::tt_metal::ccl::ShardType::Height;
                break;
            case TensorMemoryLayout::INTERLEAVED:
            case TensorMemoryLayout::SINGLE_BANK:
            default:
                TT_ASSERT(false, "ShardedAllGatherConfig only supports sharded memory layouts");
            break;
        };

        tt::tt_metal::Shape const& output_shape = output_tensor.get_legacy_shape();
        bool multiple_dims_are_multi_tile = std::count_if(output_shape.begin(), output_shape.end(), [](uint32_t s) { return s > 1; }) > 1;
        this->requires_post_all_gather_reshard = !multiple_dims_are_multi_tile;
    }

    bool get_post_all_gather_reshard_required() const {
        TT_ASSERT(is_sharding_enabled, "Tried getting sharding config for non-sharded tensor");
        return requires_post_all_gather_reshard;
    }

    bool get_single_tile_shard_on_dim() const {
        TT_ASSERT(is_sharding_enabled, "Tried getting sharding config for non-sharded tensor");
        return single_tile_shard_on_dim;
    }

    tt::tt_metal::ccl::ShardType get_shard_type() const {
        TT_ASSERT(is_sharding_enabled, "Tried getting sharding config for non-sharded tensor");
        return shard_type;
    }

    private:
    bool requires_post_all_gather_reshard;
    bool single_tile_shard_on_dim;
    tt::tt_metal::ccl::ShardType shard_type;
    bool is_sharding_enabled;
};



struct ShardAddrGenArgGenerator {
    using shard_cores_t = CoreRangeSet;

    ShardAddrGenArgGenerator(tt::tt_metal::ccl::ShardAddrGenArgs<true> const& args_struct) :
        args_struct(args_struct), initialized(true) {}

    ShardAddrGenArgGenerator() : initialized(false) {}

    std::vector<uint32_t> generate() const {
        TT_ASSERT(initialized, "Didn't initialize ShardAddrGenArgGenerator before use");
        std::vector<uint32_t> args;
        args.reserve(7 * this->args_struct.num_dest_cores * 2);

        TT_ASSERT(this->args_struct.shard_size_in_bytes != tt::tt_metal::ccl::UNINITIALIZED_VALUE_U32);
        TT_ASSERT(this->args_struct.total_chunks_per_core != tt::tt_metal::ccl::UNINITIALIZED_VALUE_U16);
        TT_ASSERT(this->args_struct.shards_start_address != tt::tt_metal::ccl::UNINITIALIZED_VALUE_U32);
        TT_ASSERT(this->args_struct.starting_core_index != tt::tt_metal::ccl::UNINITIALIZED_VALUE_U16);
        TT_ASSERT(this->args_struct.starting_chunk_into_shard != tt::tt_metal::ccl::UNINITIALIZED_VALUE_U16);
        TT_ASSERT(this->args_struct.intra_core_stride_in_shards != tt::tt_metal::ccl::UNINITIALIZED_VALUE_U16);
        TT_ASSERT(this->args_struct.contiguous_chunks_before_stride != tt::tt_metal::ccl::UNINITIALIZED_VALUE_U16);
        TT_ASSERT(this->args_struct.num_dest_cores != tt::tt_metal::ccl::UNINITIALIZED_VALUE_U16);
        TT_ASSERT(this->args_struct.dest_cores.size() != 0);

        args.push_back(this->args_struct.is_clockwise);
        args.push_back(this->args_struct.shard_size_in_bytes);
        args.push_back(this->args_struct.total_chunks_per_core);
        args.push_back(this->args_struct.shards_start_address);
        args.push_back(this->args_struct.starting_core_index);
        args.push_back(this->args_struct.starting_chunk_into_shard);
        args.push_back(this->args_struct.intra_core_stride_in_shards);
        args.push_back(this->args_struct.contiguous_chunks_before_stride);
        args.push_back(this->args_struct.num_dest_cores);
        for (tt::tt_metal::ccl::WorkerXY const& core : this->args_struct.dest_cores) {
            args.push_back(core.to_uint32());
        }

        TT_ASSERT(args.size() == args_struct.get_expected_num_args(), "Generated args size doesn't match expected size");

        return args;
    }

    void dump_to_log() const {
        log_trace(tt::LogOp, "ShardAddrGenArgGenerator:");
        log_trace(tt::LogOp, "\tis_clockwise: {}", this->args_struct.is_clockwise);
        log_trace(tt::LogOp, "\tshard_size_in_bytes: {}", this->args_struct.shard_size_in_bytes);
        log_trace(tt::LogOp, "\ttotal_chunks_per_core: {}", this->args_struct.total_chunks_per_core);
        log_trace(tt::LogOp, "\tshards_start_address: {}", this->args_struct.shards_start_address);
        log_trace(tt::LogOp, "\tstarting_core_index: {}", this->args_struct.starting_core_index);
        log_trace(tt::LogOp, "\tstarting_chunk_into_shard: {}", this->args_struct.starting_chunk_into_shard);
        log_trace(tt::LogOp, "\tintra_core_stride_in_shards: {}", this->args_struct.intra_core_stride_in_shards);
        log_trace(tt::LogOp, "\tcontiguous_chunks_before_stride: {}", this->args_struct.contiguous_chunks_before_stride);
        log_trace(tt::LogOp, "\tnum_dest_cores: {}", this->args_struct.num_dest_cores);
        for (auto n = 0; n < this->args_struct.num_dest_cores; ++n) {
            log_trace(tt::LogOp, "\t\tdest_core[{}]: x={},y={}", n, this->args_struct.dest_cores.at(n).x, this->args_struct.dest_cores.at(n).y);
        }

        TT_ASSERT(this->args_struct.dest_cores.size() == this->args_struct.num_dest_cores);
        TT_ASSERT(this->args_struct.starting_core_index < this->args_struct.num_dest_cores);
        TT_ASSERT(this->args_struct.starting_core_index < this->args_struct.dest_cores.size());
    }

    tt::tt_metal::ccl::ShardAddrGenArgs<true> args_struct;

    bool initialized;
};

struct InputTensorShardAddrGenArgGenerator final : public ShardAddrGenArgGenerator {
    static std::vector<CoreCoord> ctor_generate_dest_cores(
        CoreRangeSet const& all_shard_cores,
        uint32_t worker_index,
        uint32_t num_workers
    ) {
        bool row_wise = true;
        auto const& all_shard_cores_vec = corerange_to_cores(all_shard_cores, std::nullopt, row_wise);
        uint32_t num_shard_cores = all_shard_cores_vec.size();
        uint32_t shards_per_worker = num_shard_cores / num_workers;
        uint32_t num_dest_cores = shards_per_worker;
        bool has_extra_worker = worker_index < num_shard_cores % num_workers;
        if (has_extra_worker) {
            num_dest_cores++;
        }

        uint32_t worker_cores_start = worker_index * shards_per_worker + std::min(num_shard_cores % num_workers, worker_index);
        std::vector<CoreCoord> dest_cores;
        dest_cores.reserve(num_dest_cores);
        for (uint32_t c = worker_cores_start; c < worker_cores_start + num_dest_cores; ++c) {
            CoreCoord const& worker_core = all_shard_cores_vec.at(c);
            dest_cores.push_back(worker_core);
        }
        return dest_cores;
    }
    InputTensorShardAddrGenArgGenerator(
        Device const* device,
        tt::tt_metal::ccl::CclOpShardedTensorConfig *input_tensor_config,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t num_workers,
        uint32_t worker_index,
        // Which of the args_struct dest_cores to start at
        uint32_t starting_dest_core_index,
        uint32_t starting_chunk_into_shard,
        bool is_worker_in_clockwise_ring
        ) {
        TT_ASSERT(input_tensor_config != nullptr);
        auto const& tensor_shard_grid = input_tensor_config->get_shard_spec().grid;
        uint32_t sharded_tensor_num_cores = tensor_shard_grid.num_cores();
        this->args_struct.is_clockwise = is_worker_in_clockwise_ring;
        this->args_struct.shard_size_in_bytes = input_tensor_config->get_shard_size_in_bytes();
        this->args_struct.total_chunks_per_core = 1;
        this->args_struct.shards_start_address = input_tensor_config->get_buffer_start_address();

        this->args_struct.starting_core_index = starting_dest_core_index;
        this->args_struct.starting_chunk_into_shard = starting_chunk_into_shard;
        TT_ASSERT(sharded_tensor_num_cores > 0);

        this->args_struct.intra_core_stride_in_shards = 1;
        this->args_struct.contiguous_chunks_before_stride = 1;

        std::vector<CoreCoord> const& dest_core_coords = ctor_generate_dest_cores(
            input_tensor_config->get_shard_spec().grid,
            worker_index,
            num_workers
        );
        this->args_struct.dest_cores.reserve(dest_core_coords.size());
        std::transform(dest_core_coords.begin(), dest_core_coords.end(), std::back_inserter(this->args_struct.dest_cores),
            [&device](CoreCoord const& core) {
                return tt::tt_metal::ccl::WorkerXY(
                    static_cast<uint16_t>(device->worker_core_from_logical_core(core).x),
                    static_cast<uint16_t>(device->worker_core_from_logical_core(core).y)
                    );
            });
        TT_ASSERT(this->args_struct.dest_cores.size() > 0);

        this->args_struct.num_dest_cores = this->args_struct.dest_cores.size();
        TT_ASSERT(this->args_struct.starting_chunk_into_shard < this->args_struct.num_dest_cores);
        TT_ASSERT(this->args_struct.starting_chunk_into_shard < this->args_struct.dest_cores.size());
        TT_ASSERT(this->args_struct.dest_cores.size() == this->args_struct.num_dest_cores);

        this->initialized = true;
    }
};


struct OutputTensorShardAddrGenArgGenerator final : ShardAddrGenArgGenerator {
    static std::vector<CoreCoord> compute_worker_coord_worker_dest_cores (
        tt::tt_metal::ccl::ShardType shard_type,
        std::vector<CoreCoord> const& global_shard_dest_cores,
        uint32_t input_num_shards,
        uint32_t output_num_shards,
        uint32_t num_workers,
        uint32_t worker_index,
        bool is_shard_orientation_row_major) {
            TT_ASSERT(output_num_shards % input_num_shards == 0);

            TT_ASSERT(output_num_shards % num_workers == 0, "Don't support otherwise");
            TT_ASSERT(output_num_shards % global_shard_dest_cores.size() == 0, "Don't support otherwise");
            uint32_t const output_shards_served_per_worker = (output_num_shards / num_workers);
            uint32_t const input_shards_per_dest_core = output_num_shards / global_shard_dest_cores.size();
            TT_ASSERT(output_shards_served_per_worker % input_shards_per_dest_core == 0, "Don't support otherwise");

            uint32_t const input_shards_per_input_worker = input_num_shards / num_workers;
            TT_ASSERT(input_shards_per_dest_core > input_shards_per_input_worker ?
                input_shards_per_dest_core % input_shards_per_input_worker == 0:
                input_shards_per_input_worker / input_shards_per_dest_core > 0,
                "Don't support otherwise");
            uint32_t const contiguous_dest_cores_before_stride = input_shards_per_input_worker < input_shards_per_dest_core ? 1 : input_shards_per_input_worker / input_shards_per_dest_core;
            TT_ASSERT(contiguous_dest_cores_before_stride != 0);

            uint32_t const worker_input_shard_index = worker_index * input_shards_per_input_worker;
            uint32_t current_dest_core_offset = worker_input_shard_index / input_shards_per_dest_core;
            TT_ASSERT(current_dest_core_offset < global_shard_dest_cores.size(), "Out of bounds index generated");
            uint32_t const stride = std::max<uint32_t>(
                1,
                contiguous_dest_cores_before_stride > 1
                    ? num_workers * contiguous_dest_cores_before_stride
                    : (num_workers * input_shards_per_input_worker / input_shards_per_dest_core));

            std::vector<CoreCoord> dest_cores_of_worker;

            uint32_t core_offset_into_contiguous_chunk = 0;
            while (current_dest_core_offset + core_offset_into_contiguous_chunk < global_shard_dest_cores.size()) {
                CoreCoord const& dest_core_coord = global_shard_dest_cores.at(current_dest_core_offset + core_offset_into_contiguous_chunk);
                dest_cores_of_worker.push_back(dest_core_coord);
                core_offset_into_contiguous_chunk++;

                if (core_offset_into_contiguous_chunk == contiguous_dest_cores_before_stride) {
                    current_dest_core_offset += stride;
                    core_offset_into_contiguous_chunk = 0;
                }
            }

            return dest_cores_of_worker;
    }


    static std::vector<tt::tt_metal::ccl::WorkerXY> compute_worker_dest_cores (
        tt::tt_metal::ccl::ShardType shard_type,
        Device const& device,
        CoreRangeSet const& shard_core_range,
        uint32_t input_num_shards,
        uint32_t output_num_shards,
        uint32_t num_workers,
        uint32_t worker_index,
        bool is_shard_orientation_row_major) {
            auto const& worker_coord_worker_dest_cores = compute_worker_coord_worker_dest_cores (
                shard_type,
                corerange_to_cores(shard_core_range, std::nullopt, is_shard_orientation_row_major),
                input_num_shards,
                output_num_shards,
                num_workers,
                worker_index,
                is_shard_orientation_row_major);

            std::vector<tt::tt_metal::ccl::WorkerXY> dest_cores_of_worker;
            dest_cores_of_worker.reserve(worker_coord_worker_dest_cores.size());
            std::transform(worker_coord_worker_dest_cores.begin(), worker_coord_worker_dest_cores.end(), std::back_inserter(dest_cores_of_worker),
                [&device](CoreCoord const& core) {
                    return tt::tt_metal::ccl::WorkerXY(
                        static_cast<uint16_t>(device.worker_core_from_logical_core(core).x),
                        static_cast<uint16_t>(device.worker_core_from_logical_core(core).y)
                        );
                });
            return dest_cores_of_worker;
    }


    static std::pair<uint32_t, uint32_t> get_first_output_shard_starting_location(
        uint32_t num_workers,
        uint32_t input_tensor_shard_grid_size,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t serving_worker_index) {

        uint32_t const global_num_buffers_per_chip = num_workers;
        uint32_t const num_dest_shard_cores = input_tensor_shard_grid_size;

        uint32_t num_input_tensor_shards = input_tensor_shard_grid_size;
        uint32_t global_output_shard_index = ring_index * num_input_tensor_shards;

        TT_ASSERT(num_input_tensor_shards >= global_num_buffers_per_chip, "Not enough input shards to support all gather");
        uint32_t worker_shard_offset = serving_worker_index * (num_input_tensor_shards / global_num_buffers_per_chip);
        TT_ASSERT(num_input_tensor_shards % global_num_buffers_per_chip == 0, "Don't support this non-divisibility yet");
        if (num_input_tensor_shards % global_num_buffers_per_chip != 0) {
            worker_shard_offset += std::min(serving_worker_index, num_input_tensor_shards % global_num_buffers_per_chip);
        }
        uint32_t const num_output_tensor_shards = ring_size * num_input_tensor_shards;
        uint32_t const input_shards_per_output_worker = (num_output_tensor_shards / num_dest_shard_cores);
        global_output_shard_index += worker_shard_offset;

        uint32_t dest_worker_index = global_output_shard_index / input_shards_per_output_worker;
        uint32_t offset_chunk_in_worker = global_output_shard_index % input_shards_per_output_worker;
        return {dest_worker_index, offset_chunk_in_worker};
    }

    // returns pair of <dest_worker_idx, chunk_idx> -> These are global core index and relative chunk index
    static std::pair<uint32_t, uint32_t> get_first_output_shard_starting_location(
        AllGatherConfig const& all_gather_config,
        Tensor const& input_tensor,
        Tensor const& output_tensor,
        uint32_t ring_index,
        uint32_t serving_worker_index) {
        return get_first_output_shard_starting_location(
            all_gather_config.get_num_eth_buffers_per_edm(),
            input_tensor.shard_spec()->grid.num_cores(),
            ring_index,
            all_gather_config.get_ring_size(),
            serving_worker_index);
    }

    static uint16_t get_intra_core_stride_in_shards(uint32_t input_shard_grid_size, uint32_t num_workers, uint32_t ring_size) {
        // This function isn't generalized properly yet so it has some hardcoded behaviour. We need to generalize it
        if (input_shard_grid_size == num_workers && num_workers == ring_size) {
            return input_shard_grid_size;
        }
        auto stride = (num_workers == 1) ? 1 : (input_shard_grid_size / num_workers) + 1;
        TT_ASSERT(stride > 0, "Stride must be greater than 0");
        return stride;

    }
    static uint16_t get_contiguous_chunks_before_stride(uint32_t input_shard_grid_size, uint32_t num_workers, uint32_t ring_size) {
        auto n_contiguous = (num_workers == 1) ? 1 : input_shard_grid_size / num_workers;
        TT_ASSERT(n_contiguous > 0, "Stride must be greater than 0");
        return n_contiguous;
    }

    using shard_cores_t = std::variant<std::vector<CoreCoord>, CoreRange, CoreRangeSet>;
    OutputTensorShardAddrGenArgGenerator(
        AllGatherConfig const& all_gather_config,
        Device const* device,
        tt::tt_metal::ccl::CclOpShardedTensorConfig *input_tensor_config,
        tt::tt_metal::ccl::CclOpShardedTensorConfig *output_tensor_config,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t num_workers,
        uint32_t worker_index,
        uint32_t global_starting_dest_worker_index,
        uint32_t starting_chunk_into_shard,
        bool is_worker_in_clockwise_ring
        ) {
        bool is_shard_orientation_row_major = true; // hardcoded until the switch is flipped for col_major. Just needs test cycles and transpose when iterating dest cores

        auto const& tensor_shard_grid = input_tensor_config->get_shard_spec().grid;
        uint32_t sharded_tensor_num_cores = tensor_shard_grid.num_cores();
        TT_ASSERT(sharded_tensor_num_cores == output_tensor_config->get_shard_spec().grid.num_cores(), "Input and output tensor must have the same number of cores");
        this->args_struct.is_clockwise = is_worker_in_clockwise_ring;
        this->args_struct.shard_size_in_bytes = input_tensor_config->get_shard_size_in_bytes();
        this->args_struct.total_chunks_per_core = ring_size;
        this->args_struct.shards_start_address = output_tensor_config->get_buffer_start_address();

        this->args_struct.intra_core_stride_in_shards = get_intra_core_stride_in_shards(sharded_tensor_num_cores, num_workers, ring_size);
        this->args_struct.contiguous_chunks_before_stride = get_contiguous_chunks_before_stride(sharded_tensor_num_cores, num_workers, ring_size);

        this->args_struct.starting_chunk_into_shard = starting_chunk_into_shard;
        TT_ASSERT(sharded_tensor_num_cores > 0);

        uint32_t input_num_shards = sharded_tensor_num_cores;
        uint32_t output_num_shards = input_num_shards * ring_size;
        this->args_struct.dest_cores = OutputTensorShardAddrGenArgGenerator::compute_worker_dest_cores (
                tt::tt_metal::ccl::ShardType::Width,
                *device,
                tensor_shard_grid,
                input_num_shards,
                output_num_shards,
                num_workers,
                worker_index,
                is_shard_orientation_row_major);
        this->args_struct.num_dest_cores = this->args_struct.dest_cores.size();

        TT_ASSERT(this->args_struct.dest_cores.size() > 0);
        std::vector<CoreCoord> const& global_shard_dest_cores = corerange_to_cores(tensor_shard_grid, std::nullopt, is_shard_orientation_row_major);
        CoreCoord const& dest_core_coord = global_shard_dest_cores.at(global_starting_dest_worker_index);
        tt::tt_metal::ccl::WorkerXY noc0_starting_dest_core_xy(
            static_cast<uint16_t>(device->worker_core_from_logical_core(dest_core_coord).x),
            static_cast<uint16_t>(device->worker_core_from_logical_core(dest_core_coord).y)
            );
        auto it = std::find(this->args_struct.dest_cores.begin(), this->args_struct.dest_cores.end(), noc0_starting_dest_core_xy);
        TT_ASSERT(it != this->args_struct.dest_cores.end(), "Didn't find starting dest core in dest cores. Internal logic error");
        this->args_struct.starting_core_index = std::distance(this->args_struct.dest_cores.begin(), it);
        TT_ASSERT(this->args_struct.starting_core_index < this->args_struct.dest_cores.size());
        TT_ASSERT(this->args_struct.starting_core_index < this->args_struct.num_dest_cores);
        TT_ASSERT(this->args_struct.dest_cores.size() == this->args_struct.num_dest_cores);

        this->initialized = true;

    }

};

struct FullWorkerGridShardAddrGenArgGenerator {

    std::vector<uint32_t> generate() const {
        TT_ASSERT(initialized, "Didn't initialize ShardAddrGenArgGenerator before use");
        std::vector<uint32_t> args;
        args.reserve(12 + args_struct.total_num_cores);

        TT_ASSERT(args_struct.dest_cores.size() > 0, "dest_cores was uninitialized");
        TT_ASSERT(args_struct.tile_size_in_bytes != ccl::UNINITIALIZED_VALUE_U32, "tile_size_in_bytes was uninitialized");
        TT_ASSERT(args_struct.shards_start_address != ccl::UNINITIALIZED_VALUE_U32, "shards_start_address was uninitialized");
        TT_ASSERT(args_struct.curr_core_index != ccl::UNINITIALIZED_VALUE_U16, "curr_core_index was uninitialized");
        TT_ASSERT(args_struct.total_num_cores != ccl::UNINITIALIZED_VALUE_U16, "total_num_cores was uninitialized");
        TT_ASSERT(args_struct.curr_shard_tile_x != ccl::UNINITIALIZED_VALUE_U16, "curr_shard_tile_x was uninitialized");
        TT_ASSERT(args_struct.curr_shard_tile_y != ccl::UNINITIALIZED_VALUE_U16, "curr_shard_tile_y was uninitialized");
        TT_ASSERT(args_struct.curr_tile_index != ccl::UNINITIALIZED_VALUE_U16, "curr_tile_index was uninitialized");
        TT_ASSERT(args_struct.curr_shard != ccl::UNINITIALIZED_VALUE_U16, "curr_shard was uninitialized");
        TT_ASSERT(args_struct.input_shard_num_tiles_x != ccl::UNINITIALIZED_VALUE_U16, "input_shard_num_tiles_x was uninitialized");
        TT_ASSERT(args_struct.input_shard_num_tiles_y != ccl::UNINITIALIZED_VALUE_U16, "input_shard_num_tiles_y was uninitialized");
        TT_ASSERT(args_struct.total_shards_x != ccl::UNINITIALIZED_VALUE_U16, "total_shards_x was uninitialized");

        args.push_back(args_struct.tile_size_in_bytes);
        args.push_back(args_struct.shards_start_address);
        args.push_back(args_struct.curr_shard_tile_x);
        args.push_back(args_struct.curr_shard_tile_y);
        args.push_back(args_struct.curr_tile_index);
        args.push_back(args_struct.curr_shard);
        args.push_back(args_struct.input_shard_num_tiles_x);
        args.push_back(args_struct.input_shard_num_tiles_y);
        args.push_back(args_struct.total_shards_x);
        args.push_back(args_struct.is_clockwise);
        args.push_back(args_struct.curr_core_index);
        args.push_back(args_struct.total_num_cores);
        for (tt::tt_metal::ccl::WorkerXY const& core : args_struct.dest_cores) {
            args.push_back(core.to_uint32());
        }

        TT_ASSERT(args.size() == args_struct.get_expected_num_args(), "Generated args size doesn't match expected size");

        return args;
    }

    FullWorkerGridShardAddrGenArgGenerator(
        AllGatherConfig const& all_gather_config,
        Device const* device,
        Tensor const& input_tensor,
        Tensor const& output_tensor,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t worker_index,
        uint32_t starting_dest_core_index,
        uint32_t starting_tile_index,
        bool is_worker_in_clockwise_ring
        ) {
            bool is_shard_orientation_row_major = true;
            auto *input_buffer = input_tensor.buffer();
            this->args_struct.tile_size_in_bytes = input_buffer->page_size();
            this->args_struct.shards_start_address = output_tensor.buffer()->address();
            this->args_struct.curr_shard_tile_x = 0;
            this->args_struct.curr_shard_tile_y = 0;
            this->args_struct.curr_tile_index = starting_tile_index;
            this->args_struct.curr_shard = ring_index;
            this->args_struct.input_shard_num_tiles_x = input_buffer->shard_spec().tensor2d_shape[1];
            this->args_struct.input_shard_num_tiles_y = input_buffer->shard_spec().tensor2d_shape[0];
            this->args_struct.total_shards_x = ring_size;
            this->args_struct.is_clockwise = is_worker_in_clockwise_ring;

            this->args_struct.curr_core_index = starting_dest_core_index;

            auto const& tensor_shard_grid = input_tensor.buffer()->shard_spec().grid();
            this->args_struct.dest_cores = OutputTensorShardAddrGenArgGenerator::compute_worker_dest_cores (
                    tt::tt_metal::ccl::ShardType::Width,
                    *device,
                    tensor_shard_grid,
                    tensor_shard_grid.num_cores(),
                    ring_size * tensor_shard_grid.num_cores(),
                    tensor_shard_grid.num_cores(),
                    worker_index,
                    is_shard_orientation_row_major);
            this->args_struct.total_num_cores = this->args_struct.dest_cores.size();

            this->initialized = true;
        }

    tt::tt_metal::ccl::FullWorkerGridShardAddrGenArgs<true> args_struct;
    bool initialized;
};

namespace operations {
namespace ccl {

Tensor all_gather(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

} // namespace ccl
} // namespace operations

}  // namespace ttnn
