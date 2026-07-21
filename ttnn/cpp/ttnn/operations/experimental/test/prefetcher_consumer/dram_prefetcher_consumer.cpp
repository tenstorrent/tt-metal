// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_consumer.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace ttnn::operations::experimental::test {

namespace {
constexpr uint32_t kRemoteCBId = 31;
}  // namespace

void DramPrefetcherConsumerDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(attrs.mesh_device != nullptr, "mesh_device required");
    TT_FATAL(attrs.num_iters > 0, "num_iters must be > 0");
    TT_FATAL(attrs.page_size_bytes > 0, "page_size_bytes must be > 0");
    TT_FATAL(attrs.global_cb.has_value(), "global_cb required");
    TT_FATAL(attrs.global_cb->receiver_cores().num_cores() > 0, "GCB has no receiver cores");
}

void DramPrefetcherConsumerDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

DramPrefetcherConsumerDeviceOperation::spec_return_value_t DramPrefetcherConsumerDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t&) {
    return std::vector<ttnn::TensorSpec>{};
}

DramPrefetcherConsumerDeviceOperation::tensor_return_value_t
DramPrefetcherConsumerDeviceOperation::create_output_tensors(const operation_attributes_t&, const tensor_args_t&) {
    return std::vector<ttnn::Tensor>{};
}

ttsl::hash::hash_t DramPrefetcherConsumerDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/) {
    // GlobalCircularBuffer isn't reflection-hashable; hash its identity via config_address
    // (unique per GCB instance on this device) along with the other attrs.
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<DramPrefetcherConsumerDeviceOperation>,
        attrs.num_iters,
        attrs.page_size_bytes,
        static_cast<uint64_t>(attrs.global_cb->config_address()));
}

ttnn::device_operation::CachedProgram<DramPrefetcherConsumerDeviceOperation::ProgramFactory::shared_variables_t>
DramPrefetcherConsumerDeviceOperation::ProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const tensor_args_t& /*tensor_args*/,
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

    const std::vector<uint32_t> compile_args = {kRemoteCBId, operation_attributes.num_iters};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_bench_discard_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

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
        .global_cb = global_cb,
        .mesh_device = mesh_device,
    };
    OperationType::tensor_args_t tensor_args{};
    ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::operations::experimental::test
