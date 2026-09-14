// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_consumer.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <functional>

namespace ttnn::operations::experimental::test {

namespace {
constexpr uint32_t kRemoteCBId = 31;
constexpr uint32_t kOrdinaryReadScratchCBId = 0;
}  // namespace

void DramPrefetcherConsumerDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(attrs.mesh_device != nullptr, "mesh_device required");
    TT_FATAL(attrs.num_iters > 0, "num_iters must be > 0");
    TT_FATAL(attrs.page_size_bytes > 0, "page_size_bytes must be > 0");
    TT_FATAL(attrs.global_cb.has_value(), "global_cb required");
    TT_FATAL(attrs.global_cb->receiver_cores().num_cores() > 0, "GCB has no receiver cores");
    if (attrs.ordinary_read_bytes > 0) {
        const uint32_t dram_alignment =
            attrs.mesh_device->allocator()->get_alignment(tt::tt_metal::BufferType::DRAM);
        TT_FATAL(
            attrs.ordinary_read_bytes % dram_alignment == 0,
            "ordinary_read_bytes must be DRAM-aligned");
        TT_FATAL(tensor_args.ordinary_source_tensor.has_value(), "ordinary_source_tensor required");
        const auto* source_buffer = tensor_args.ordinary_source_tensor->buffer();
        TT_FATAL(source_buffer != nullptr && source_buffer->is_dram(), "ordinary_source_tensor must be in DRAM");
        TT_FATAL(
            source_buffer->aligned_size_per_bank() >= attrs.ordinary_read_bytes,
            "ordinary source allocation has {} bytes per bank, smaller than requested {}-byte read",
            source_buffer->aligned_size_per_bank(),
            attrs.ordinary_read_bytes);
        TT_FATAL(tensor_args.timing_tensor.has_value(), "timing_tensor required");
        const auto* timing_buffer = tensor_args.timing_tensor->buffer();
        TT_FATAL(timing_buffer != nullptr && timing_buffer->is_l1(), "timing_tensor must be in L1");
        TT_FATAL(timing_buffer->aligned_page_size() >= 8 * sizeof(uint32_t), "timing tensor shards must hold 8 words");
    }
}

void DramPrefetcherConsumerDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

DramPrefetcherConsumerDeviceOperation::spec_return_value_t DramPrefetcherConsumerDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t&) {
    return std::vector<tt::tt_metal::TensorSpec>{};
}

DramPrefetcherConsumerDeviceOperation::tensor_return_value_t
DramPrefetcherConsumerDeviceOperation::create_output_tensors(const operation_attributes_t&, const tensor_args_t&) {
    return std::vector<ttnn::Tensor>{};
}

ttsl::hash::hash_t DramPrefetcherConsumerDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto* source_buffer =
        tensor_args.ordinary_source_tensor.has_value() ? tensor_args.ordinary_source_tensor->buffer() : nullptr;
    const auto* timing_buffer = tensor_args.timing_tensor.has_value() ? tensor_args.timing_tensor->buffer() : nullptr;
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<DramPrefetcherConsumerDeviceOperation>,
        attrs.num_iters,
        attrs.page_size_bytes,
        attrs.ordinary_read_bytes,
        std::hash<tt::tt_metal::experimental::GlobalCircularBuffer>{}(*attrs.global_cb),
        static_cast<uint64_t>(attrs.global_cb->buffer_address()),
        static_cast<uint64_t>(attrs.global_cb->config_address()),
        static_cast<uint64_t>(source_buffer != nullptr ? source_buffer->address() : 0),
        static_cast<uint64_t>(timing_buffer != nullptr ? timing_buffer->address() : 0));
}

ttnn::device_operation::CachedProgram<DramPrefetcherConsumerDeviceOperation::ProgramFactory::shared_variables_t>
DramPrefetcherConsumerDeviceOperation::ProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    using namespace tt::tt_metal;

    Program program = CreateProgram();
    const auto& global_cb = operation_attributes.global_cb.value();
    const CoreRangeSet receiver_cores = global_cb.receiver_cores();

    // Configure the receiver-side CB. set_page_size matches what the sender resizes the CB to
    // (in_block_w_tiles * n_tiles_per_recv * tile_bytes); receiver wait_front/pop_front operate
    // in units of this page size.
    CircularBufferConfig cb_config(operation_attributes.page_size_bytes);
    cb_config.remote_index(kRemoteCBId)
        .set_page_size(operation_attributes.page_size_bytes)
        .set_data_format(tt::DataFormat::Float16_b);
    tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, cb_config, global_cb);

    if (operation_attributes.ordinary_read_bytes > 0) {
        CircularBufferConfig scratch_config(
            operation_attributes.ordinary_read_bytes,
            {{kOrdinaryReadScratchCBId, tt::DataFormat::Float16_b}});
        scratch_config.set_page_size(kOrdinaryReadScratchCBId, operation_attributes.ordinary_read_bytes);
        CreateCircularBuffer(program, receiver_cores, scratch_config);
    }

    const std::vector<uint32_t> compile_args = {
        kRemoteCBId,
        operation_attributes.num_iters,
        operation_attributes.ordinary_read_bytes,
        kOrdinaryReadScratchCBId,
    };
    const KernelHandle kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_bench_discard_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

    if (operation_attributes.ordinary_read_bytes > 0) {
        const uint32_t source_address =
            static_cast<uint32_t>(tensor_args.ordinary_source_tensor.value().buffer()->address());
        const uint32_t timing_address = static_cast<uint32_t>(tensor_args.timing_tensor.value().buffer()->address());
        for (const auto& [sender_logical, receivers] : global_cb.sender_receiver_core_mapping()) {
            const uint32_t bank_id = sender_logical.x;
            for (const CoreCoord& receiver : corerange_to_cores(receivers, std::nullopt, /*row_wise=*/true)) {
                SetRuntimeArgs(program, kernel_id, receiver, {bank_id, source_address, timing_address});
            }
        }
    }

    return {std::move(program), shared_variables_t{}};
}

void DramPrefetcherConsumerDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& /*cached_workload*/,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // Nothing to override — all args are compile-time.
}

void test_dram_prefetcher_consumer(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb) {
    using OperationType = DramPrefetcherConsumerDeviceOperation;
    OperationType::operation_attributes_t attrs{
        .num_iters = num_iters,
        .page_size_bytes = page_size_bytes,
        .ordinary_read_bytes = 0,
        .global_cb = global_cb,
        .mesh_device = mesh_device,
    };
    OperationType::tensor_args_t tensor_args{
        .ordinary_source_tensor = std::nullopt,
        .timing_tensor = std::nullopt,
    };
    ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

void test_dram_prefetcher_contention_consumer(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::Tensor& ordinary_source_tensor,
    const ttnn::Tensor& timing_tensor,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    uint32_t ordinary_read_bytes,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb) {
    using OperationType = DramPrefetcherConsumerDeviceOperation;
    OperationType::operation_attributes_t attrs{
        .num_iters = num_iters,
        .page_size_bytes = page_size_bytes,
        .ordinary_read_bytes = ordinary_read_bytes,
        .global_cb = global_cb,
        .mesh_device = mesh_device,
    };
    OperationType::tensor_args_t tensor_args{
        .ordinary_source_tensor = ordinary_source_tensor,
        .timing_tensor = timing_tensor,
    };
    ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::operations::experimental::test
