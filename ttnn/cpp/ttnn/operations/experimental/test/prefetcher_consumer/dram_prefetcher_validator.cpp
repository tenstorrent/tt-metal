// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_validator.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <unordered_map>

namespace ttnn::operations::experimental::test {

namespace {
constexpr uint32_t kValidatorRemoteCBId = 31;
constexpr uint32_t kValidatorScratchCBId = 0;
}  // namespace

void DramPrefetcherValidatorDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(attrs.num_layers > 0, "num_layers must be > 0");
    const auto* tensor_buffer = tensor_args.source_tensor.buffer();
    TT_FATAL(tensor_buffer != nullptr, "source_tensor must be on device");
    TT_FATAL(tensor_buffer->is_dram(), "source_tensor must be a DRAM buffer");
    TT_FATAL(attrs.global_cb.has_value(), "global_cb required");
    TT_FATAL(attrs.global_cb->receiver_cores().num_cores() > 0, "GCB has no receiver cores");

    const auto& sr_mapping = attrs.global_cb->sender_receiver_core_mapping();
    TT_FATAL(!sr_mapping.empty(), "GCB has no senders");
}

void DramPrefetcherValidatorDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

DramPrefetcherValidatorDeviceOperation::spec_return_value_t
DramPrefetcherValidatorDeviceOperation::compute_output_specs(const operation_attributes_t&, const tensor_args_t&) {
    return std::vector<ttnn::TensorSpec>{};
}

DramPrefetcherValidatorDeviceOperation::tensor_return_value_t
DramPrefetcherValidatorDeviceOperation::create_output_tensors(const operation_attributes_t&, const tensor_args_t&) {
    return std::vector<ttnn::Tensor>{};
}

ttsl::hash::hash_t DramPrefetcherValidatorDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // GlobalCircularBuffer / Tensor aren't reflection-hashable here; pick the bits that
    // determine Program shape: scalar attrs, GCB identity, the source tensor's DRAM
    // address (compile-time arg via TensorAccessorArgs), and its dataformat.
    const auto* tensor_buffer = tensor_args.source_tensor.buffer();
    const tt::DataFormat dataformat = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.source_tensor.dtype());
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<DramPrefetcherValidatorDeviceOperation>,
        attrs.num_layers,
        attrs.print_stride,
        attrs.streaming,
        attrs.rotation,
        static_cast<uint64_t>(attrs.global_cb->config_address()),
        static_cast<uint64_t>(tensor_buffer != nullptr ? tensor_buffer->address() : 0),
        static_cast<uint32_t>(dataformat));
}

ttnn::device_operation::CachedProgram<DramPrefetcherValidatorDeviceOperation::ProgramFactory::shared_variables_t>
DramPrefetcherValidatorDeviceOperation::ProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    using namespace tt::tt_metal;

    const auto& source_tensor = tensor_args.source_tensor;
    const auto& global_cb = operation_attributes.global_cb.value();

    // Derive ring topology from the GCB (matches both worker-sender and DRAM-sender paths).
    const auto& sr_mapping = global_cb.sender_receiver_core_mapping();
    const uint32_t num_senders = static_cast<uint32_t>(sr_mapping.size());
    uint32_t num_blocks = 0;
    uint32_t max_bank_id = 0;
    for (const auto& [sender_logical, receivers] : sr_mapping) {
        const uint32_t bank_id = static_cast<uint32_t>(sender_logical.x);
        max_bank_id = bank_id > max_bank_id ? bank_id : max_bank_id;
        num_blocks += receivers.num_cores();
    }
    const uint32_t num_dram_banks = max_bank_id + 1;
    TT_FATAL(
        num_dram_banks > 0 && num_blocks % num_dram_banks == 0,
        "Validator: total receiver count ({}) must divide evenly across {} DRAM banks",
        num_blocks,
        num_dram_banks);
    const uint32_t receivers_per_bank = num_blocks / num_dram_banks;

    // Per-tensor geometry (single-tensor path; see prefetcher_matmul_design.md §3).
    // Read shape from the tensor's logical (padded) shape so this works for
    // both legacy WIDTH_SHARDED (one shard per bank) and ND_SHARDED
    // (num_shards = ring_size, receiver-contiguous) layouts. The
    // per-(ring_pos, block) tile mapping is layout-mode-invariant; only the
    // ring_pos -> (bank, recv_idx) formula differs.
    Buffer* tensor_buffer = source_tensor.buffer();
    const auto& padded_shape = source_tensor.padded_shape();
    TT_FATAL(
        padded_shape.rank() >= 2,
        "Validator: source tensor padded shape must be at least rank 2; got rank {}",
        padded_shape.rank());
    const auto& tile_spec = source_tensor.tensor_spec().tile();
    const auto tile_shape = tile_spec.get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t K_elems = padded_shape[-2];
    const uint32_t N_elems = padded_shape[-1];
    TT_FATAL(
        K_elems % tile_h == 0 && N_elems % tile_w == 0,
        "Validator: tensor padded shape ({}, {}) must be tile-aligned (tile {}x{})",
        K_elems,
        N_elems,
        tile_h,
        tile_w);
    const uint32_t k_tiles = K_elems / tile_h;
    const uint32_t total_n_tiles = N_elems / tile_w;
    TT_FATAL(
        k_tiles % num_blocks == 0, "Validator: k_tiles ({}) must be divisible by num_blocks ({})", k_tiles, num_blocks);
    const uint32_t ring_size = num_blocks;
    TT_FATAL(
        total_n_tiles % ring_size == 0,
        "Validator: total_n_tiles ({}) must be divisible by ring_size ({})",
        total_n_tiles,
        ring_size);
    const uint32_t n_per_recv_tiles = total_n_tiles / ring_size;
    const uint32_t k_block_w_tiles = k_tiles / num_blocks;
    const tt::DataFormat tensor_dataformat = datatype_to_dataformat_converter(source_tensor.dtype());
    const uint32_t tile_bytes = tile_spec.get_tile_size(tensor_dataformat);
    const uint32_t page_bytes_per_recv = k_block_w_tiles * n_per_recv_tiles * tile_bytes;

    // Layout detection: ND_SHARDED tensors with `num_shards == ring_size` are
    // the receiver-contiguous DRAM-core layout. Under ROUND_ROBIN_1D shard
    // distribution, the natural GCB pairing is strided (bank b feeds ring
    // positions b, b + num_senders, ...). Under CONTIGUOUS_1D (shard-contiguous) shard
    // distribution, bank b owns a contiguous run of shards, so it feeds the
    // contiguous ring arc b*R .. b*R+R-1 (same pairing as legacy WIDTH_SHARDED).
    const bool is_recv_contig =
        source_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::ND_SHARDED &&
        tensor_buffer->buffer_distribution_spec().has_value() &&
        tensor_buffer->buffer_distribution_spec()->num_shards() == ring_size;
    const bool is_shard_contiguous_recv_contig =
        is_recv_contig && tensor_buffer->buffer_distribution_spec()->shard_distribution_strategy() ==
                              tt::tt_metal::ShardDistributionStrategy::CONTIGUOUS_1D;

    const CoreRangeSet receiver_cores = global_cb.receiver_cores();

    Program program = CreateProgram();

    // Receiver-side remote CB: wait_front/pop_front units are one full per-receiver block.
    CircularBufferConfig remote_cfg(page_bytes_per_recv);
    remote_cfg.remote_index(kValidatorRemoteCBId).set_page_size(page_bytes_per_recv).set_data_format(tensor_dataformat);
    tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, remote_cfg, global_cb);

    // Scratch CB: holds the expected page bytes during a single block comparison.
    CircularBufferConfig scratch_cfg(page_bytes_per_recv, {{kValidatorScratchCBId, tensor_dataformat}});
    scratch_cfg.set_page_size(kValidatorScratchCBId, page_bytes_per_recv);
    CreateCircularBuffer(program, receiver_cores, scratch_cfg);

    // Compile-time args: scalars, then the TensorAccessor args for the source tensor.
    std::vector<uint32_t> compile_args = {
        kValidatorRemoteCBId,
        kValidatorScratchCBId,
        operation_attributes.num_layers,
        num_blocks,
        num_senders,
        operation_attributes.print_stride,
        operation_attributes.streaming ? 1u : 0u,
    };
    TensorAccessorArgs(*tensor_buffer).append_to(compile_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_validator_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

    // Per-receiver runtime args: bank_id, recv_idx_in_bank, then per-tensor geometry.
    // Receiver enumeration order within a sender's CoreRangeSet must match the order the
    // sender uses to address them in its internal noc_xy table (row-major).
    const uint32_t bank_base_addr = static_cast<uint32_t>(tensor_buffer->address());
    std::unordered_map<uint32_t, uint32_t> receivers_seen_by_bank;
    for (uint32_t s = 0; s < num_senders; ++s) {
        const auto& [sender_logical, receivers_set] = sr_mapping[s];
        const uint32_t bank_id = sender_logical.x;
        const auto recv_cores = corerange_to_cores(receivers_set, std::nullopt, /*row_wise=*/true);
        const uint32_t recv_index_base = receivers_seen_by_bank[bank_id];
        receivers_seen_by_bank[bank_id] = recv_index_base + static_cast<uint32_t>(recv_cores.size());
        for (uint32_t r = 0; r < recv_cores.size(); ++r) {
            const uint32_t bank_local_recv = recv_index_base + r;
            TT_FATAL(
                bank_local_recv < receivers_per_bank,
                "Sender {} on bank {} maps receiver {} past receivers_per_bank {}",
                s,
                bank_id,
                bank_local_recv,
                receivers_per_bank);
            // ring_pos formula differs by layout:
            //   round-robin recv-contig + strided GCB: bank b's slot k -> ring_pos = b + k * num_dram_banks
            //   shard-contiguous recv-contig and legacy K-row-major: bank b's slot k -> ring_pos = b *
            //   receivers_per_bank + k
            // With dual senders, k is the bank-local receiver index across both senders.
            const bool strided_pairing = is_recv_contig && !is_shard_contiguous_recv_contig;
            const uint32_t ring_pos = strided_pairing ? (bank_id + bank_local_recv * num_dram_banks)
                                                      : (bank_id * receivers_per_bank + bank_local_recv);
            const uint32_t n_col_start = ring_pos * n_per_recv_tiles;
            // Lead physical block this receiver expects at FIFO position 0 under streaming:
            // rotation[ring_pos] when a rotation was supplied (must match the prefetcher), else
            // ring_pos for the identity (natural topology) order.
            uint32_t lead_block = ring_pos;
            if (!operation_attributes.rotation.empty()) {
                TT_FATAL(
                    ring_pos < operation_attributes.rotation.size(),
                    "Validator rotation has {} entries but ring_pos {} indexes past it",
                    operation_attributes.rotation.size(),
                    ring_pos);
                lead_block = operation_attributes.rotation[ring_pos];
            }
            std::vector<uint32_t> rt_args = {
                bank_id,
                bank_local_recv,
                bank_base_addr,
                k_block_w_tiles,
                total_n_tiles,
                n_per_recv_tiles,
                n_col_start,
                lead_block,
            };
            SetRuntimeArgs(program, kernel_id, recv_cores[r], rt_args);
        }
    }

    return {std::move(program), shared_variables_t{}};
}

void DramPrefetcherValidatorDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& /*cached_workload*/,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // Nothing to override — buffer address is part of the cache key via compute_program_hash.
}

void test_dram_prefetcher_validator(
    tt::tt_metal::distributed::MeshDevice* /*mesh_device*/,
    const ttnn::Tensor& source_tensor,
    uint32_t num_layers,
    uint32_t print_stride,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    bool streaming,
    const std::vector<uint32_t>& rotation) {
    using OperationType = DramPrefetcherValidatorDeviceOperation;
    OperationType::operation_attributes_t attrs{
        .num_layers = num_layers,
        .print_stride = print_stride,
        .global_cb = global_cb,
        .streaming = streaming,
        .rotation = rotation,
    };
    OperationType::tensor_args_t tensor_args{.source_tensor = source_tensor};
    ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::operations::experimental::test
