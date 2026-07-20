// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include <ctime>
#include <limits>
#include <memory>
#include <random>

namespace ttnn::operations::rand {

namespace {
// seed == 0 means "nondeterministic". Draw a concrete random base seed ONCE here, on the host, so the
// device op is always deterministic given its attributes: override_runtime_arguments and the
// descriptor-patching parity rebuild then derive identical per-core seeds from that base. A fresh base
// is drawn per call, so seed == 0 still yields new data on every invocation.
uint32_t resolve_seed(uint32_t seed) {
    if (seed != 0) {
        return seed;
    }
    // thread_local: seed==0 dispatches from different host threads must not race on the engine.
    thread_local std::mt19937 rng(static_cast<std::mt19937::result_type>(std::time(nullptr)));
    thread_local std::uniform_int_distribution<uint32_t> dist(1, std::numeric_limits<int32_t>::max());
    return dist(rng);
}
}  // namespace

void RandDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(
        operation_attributes.from < operation_attributes.to, "Rand: `from` argument must be < `to` argument");
}

void RandDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

TensorSpec RandDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    return ttnn::TensorSpec(
        operation_attributes.shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
}

RandDeviceOperation::tensor_return_value_t RandDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    return create_device_tensor(
        ttnn::TensorSpec(
            operation_attributes.shape,
            tt::tt_metal::TensorLayout(
                operation_attributes.dtype,
                tt::tt_metal::PageConfig(operation_attributes.layout),
                operation_attributes.memory_config)),
        operation_attributes.device);
}

}  // namespace ttnn::operations::rand

namespace ttnn::prim {
ttnn::operations::rand::RandDeviceOperation::tensor_return_value_t uniform(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    float from,
    float to,
    uint32_t seed,
    ttsl::SmallVector<bool> mesh_dim_is_sharded) {
    using OperationType = ttnn::operations::rand::RandDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .shape = shape,
            .dtype = dtype,
            .layout = layout,
            .memory_config = memory_config,
            .device = std::addressof(device),
            .from = from,
            .to = to,
            .seed = ttnn::operations::rand::resolve_seed(seed),
            .mesh_dim_is_sharded = std::move(mesh_dim_is_sharded)},
        OperationType::tensor_args_t{});
}
}  // namespace ttnn::prim
