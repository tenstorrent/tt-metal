// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_validator.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/buffer.hpp>
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
    const uint32_t num_receivers_per_sender = sr_mapping.front().second.num_cores();
    for (const auto& [_, recv] : sr_mapping) {
        TT_FATAL(
            recv.num_cores() == num_receivers_per_sender,
            "GCB has non-uniform receivers per sender ({} vs {}); validator requires uniform fanout",
            recv.num_cores(),
            num_receivers_per_sender);
    }
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

tt::stl::hash::hash_t DramPrefetcherValidatorDeviceOperation::compute_program_hash(
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
        static_cast<uint64_t>(attrs.global_cb->config_address()),
        static_cast<uint64_t>(tensor_buffer != nullptr ? tensor_buffer->address() : 0),
        static_cast<uint32_t>(dataformat));
}

ttnn::device_operation::CachedProgram<DramPrefetcherValidatorDeviceOperation::ProgramFactory::shared_variables_t>
DramPrefetcherValidatorDeviceOperation::ProgramFactory::create_at(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    using namespace tt::tt_metal;

    const auto& source_tensor = tensor_args.source_tensor;
    const auto& global_cb = attrs.global_cb.value();

    // Derive ring topology from the GCB (matches both worker-sender and DRAM-sender paths).
    const auto& sr_mapping = global_cb.sender_receiver_core_mapping();
    const uint32_t num_senders = static_cast<uint32_t>(sr_mapping.size());
    const uint32_t num_receivers_per_sender = sr_mapping.front().second.num_cores();
    const uint32_t num_blocks = num_senders * num_receivers_per_sender;

    // Per-tensor geometry (single-tensor path; see prefetcher_matmul_design.md §3).
    Buffer* tensor_buffer = source_tensor.buffer();
    const auto shard_shape = tensor_buffer->shard_spec().shape();
    const auto& tile_spec = source_tensor.tensor_spec().tile();
    const auto tile_shape = tile_spec.get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    TT_FATAL(
        shard_shape[0] % tile_h == 0 && shard_shape[1] % tile_w == 0,
        "shard_shape ({}, {}) must be tile-aligned (tile {}x{})",
        shard_shape[0],
        shard_shape[1],
        tile_h,
        tile_w);
    const uint32_t k_tiles = shard_shape[0] / tile_h;
    const uint32_t n_per_bank_tiles = shard_shape[1] / tile_w;
    TT_FATAL(
        k_tiles % num_blocks == 0,
        "k_tiles ({}) must be divisible by num_blocks ({}) for the validator's expected layout",
        k_tiles,
        num_blocks);
    TT_FATAL(
        n_per_bank_tiles % num_receivers_per_sender == 0,
        "n_per_bank_tiles ({}) must be divisible by num_receivers_per_sender ({})",
        n_per_bank_tiles,
        num_receivers_per_sender);
    const uint32_t k_block_w_tiles = k_tiles / num_blocks;
    const uint32_t n_per_recv_tiles = n_per_bank_tiles / num_receivers_per_sender;
    const tt::DataFormat tensor_dataformat = datatype_to_dataformat_converter(source_tensor.dtype());
    const uint32_t tile_bytes = tile_spec.get_tile_size(tensor_dataformat);
    const uint32_t page_bytes_per_recv = k_block_w_tiles * n_per_recv_tiles * tile_bytes;

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
        attrs.num_layers,
        num_blocks,
        num_senders,
        attrs.print_stride,
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
    for (uint32_t s = 0; s < num_senders; ++s) {
        const auto& [sender_logical, receivers_set] = sr_mapping[s];
        const uint32_t bank_id = sender_logical.x;
        const auto recv_cores = corerange_to_cores(receivers_set, std::nullopt, /*row_wise=*/true);
        TT_FATAL(
            recv_cores.size() == num_receivers_per_sender,
            "Sender {} has {} receivers but expected {}",
            s,
            recv_cores.size(),
            num_receivers_per_sender);
        for (uint32_t r = 0; r < recv_cores.size(); ++r) {
            std::vector<uint32_t> rt_args = {
                bank_id,
                r,
                bank_base_addr,
                k_block_w_tiles,
                n_per_bank_tiles,
                n_per_recv_tiles,
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
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb) {
    using OperationType = DramPrefetcherValidatorDeviceOperation;
    OperationType::operation_attributes_t attrs{
        .num_layers = num_layers,
        .print_stride = print_stride,
        .global_cb = global_cb,
    };
    OperationType::tensor_args_t tensor_args{.source_tensor = source_tensor};
    ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::operations::experimental::test
