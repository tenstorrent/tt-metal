// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_device_operation.hpp"

#include <tt-metalium/tt_metal.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

AllGatherRegimeAMatmulAsyncDeviceOperation::program_factory_t
AllGatherRegimeAMatmulAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return AllGatherRegimeAMatmulAsyncProgramFactory{};
}

void AllGatherRegimeAMatmulAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight_tensor;
    TT_FATAL(
        act.storage_type() == StorageType::DEVICE && weight.storage_type() == StorageType::DEVICE,
        "all_gather_regime_a_matmul_async operands must be on device");
    TT_FATAL(
        act.layout() == Layout::TILE && weight.layout() == Layout::TILE,
        "all_gather_regime_a_matmul_async requires TILE layout");
    TT_FATAL(
        act.dtype() == DataType::BFLOAT16 && weight.dtype() == DataType::BFLOAT16,
        "all_gather_regime_a_matmul_async v1 supports only BFLOAT16 inputs");
    const auto& a = act.logical_shape();
    const auto& w = weight.logical_shape();
    const uint32_t K_local = a[-1];
    const uint32_t K_global = w[-2];
    TT_FATAL(K_local > 0 && K_global % K_local == 0, "in1 K ({}) must be a multiple of in0 K ({})", K_global, K_local);
    TT_FATAL(operation_attributes.d == K_global / K_local, "D mismatch with K-shard ratio");
    TT_FATAL(operation_attributes.d > 1, "D must be > 1 here (D=1 delegates to regime_a_matmul)");
    // in1 must be DRAM WIDTH_SHARDED across 8 banks (regime_a weight layout).
    const auto& wm = weight.memory_config();
    TT_FATAL(
        wm.buffer_type() == BufferType::DRAM && wm.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "all_gather_regime_a_matmul_async weight must be DRAM WIDTH_SHARDED (regime_a layout)");
}

AllGatherRegimeAMatmulAsyncDeviceOperation::spec_return_value_t
AllGatherRegimeAMatmulAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight_tensor;
    const uint32_t N = weight.logical_shape()[-1];
    const uint32_t K_global = weight.logical_shape()[-2];
    const auto dtype = operation_attributes.output_dtype.value_or(DataType::BFLOAT16);
    const auto mem = operation_attributes.output_mem_config.value_or(MemoryConfig{});
    ttnn::Shape out_shape(act.logical_shape());
    out_shape[-1] = N;  // [.., M, N]

    // Slot 1: the per-device DRAM gather buffer [.., M, K_global] (interleaved bf16). Allocated as a mesh
    // tensor so it lives at the SAME address on every device — the fabric injector writes remote shards by
    // reusing this device's TensorAccessor addresses (valid on the neighbour because the address matches).
    ttnn::Shape gather_shape(act.logical_shape());
    gather_shape[-1] = K_global;  // [.., M, K_global]
    const MemoryConfig gather_mem(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    return {
        TensorSpec(out_shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem)),
        TensorSpec(gather_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), gather_mem))};
}

AllGatherRegimeAMatmulAsyncDeviceOperation::tensor_return_value_t
AllGatherRegimeAMatmulAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& s : specs) {
        outs.push_back(create_device_tensor(s, device));
    }
    return outs;
}

std::tuple<
    AllGatherRegimeAMatmulAsyncDeviceOperation::operation_attributes_t,
    AllGatherRegimeAMatmulAsyncDeviceOperation::tensor_args_t>
AllGatherRegimeAMatmulAsyncDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const RegimeAMatmulConfig>& config,
    uint32_t d,
    std::optional<uint32_t> cluster_axis,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel,
    std::vector<GlobalSemaphore> multi_device_global_semaphore,
    std::optional<GlobalSemaphore> barrier_semaphore,
    std::optional<Tensor> persistent_output_buffer,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    const auto arch = input_tensor.device()->arch();
    auto ckc = init_device_compute_kernel_config(
        arch, compute_kernel_config, MathFidelity::HiFi2, false, true, true);
    return {
        operation_attributes_t{
            .regime_a_config = config,
            .d = d,
            .cluster_axis = cluster_axis,
            .topology = topology,
            .num_links = num_links,
            .num_workers_per_link = num_workers_per_link,
            .num_buffers_per_channel = num_buffers_per_channel,
            .output_mem_config = memory_config,
            .output_dtype = dtype,
            .compute_kernel_config = ckc},
        tensor_args_t{
            .input_tensor = input_tensor,
            .weight_tensor = weight_tensor,
            .multi_device_global_semaphore = std::move(multi_device_global_semaphore),
            .barrier_semaphore = std::move(barrier_semaphore),
            .persistent_output_buffer = std::move(persistent_output_buffer)}};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> all_gather_regime_a_matmul_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const experimental::prim::RegimeAMatmulConfig>& config,
    uint32_t d,
    std::optional<uint32_t> cluster_axis,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel,
    std::vector<GlobalSemaphore> multi_device_global_semaphore,
    std::optional<GlobalSemaphore> barrier_semaphore,
    std::optional<Tensor> persistent_output_buffer,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = experimental::prim::AllGatherRegimeAMatmulAsyncDeviceOperation;
    auto [attributes, tensor_args] = OperationType::invoke(
        input_tensor,
        weight_tensor,
        config,
        d,
        cluster_axis,
        topology,
        num_links,
        num_workers_per_link,
        num_buffers_per_channel,
        std::move(multi_device_global_semaphore),
        std::move(barrier_semaphore),
        std::move(persistent_output_buffer),
        memory_config,
        dtype,
        compute_kernel_config);
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
