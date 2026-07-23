// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_device_operation.hpp"

#include <cstdlib>
#include <string>

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
    // Kernel arrays are fixed-size (shard_landed[8], compute-core coords[128]); enforce those bounds.
    TT_FATAL(operation_attributes.d <= 8, "all_gather_regime_a_matmul_async supports D<=8 (got {})", operation_attributes.d);
    // in1 must be DRAM WIDTH_SHARDED across 8 banks (regime_a weight layout).
    const auto& wm = weight.memory_config();
    TT_FATAL(
        wm.buffer_type() == BufferType::DRAM && wm.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "all_gather_regime_a_matmul_async weight must be DRAM WIDTH_SHARDED (regime_a layout)");

    // Output config: the writer emits bf16 (2 KiB) TILE tiles into an interleaved output. The public API accepts
    // other dtypes/layouts, but the kernels hardcode bf16/HiFi2-fp32acc, so reject anything else rather than
    // silently corrupting (e.g. an fp32 output would allocate fp32 storage while the writer emits bf16 tiles).
    if (operation_attributes.output_dtype.has_value()) {
        TT_FATAL(
            operation_attributes.output_dtype.value() == DataType::BFLOAT16,
            "all_gather_regime_a_matmul_async v1 only produces BFLOAT16 output (got {})",
            static_cast<int>(operation_attributes.output_dtype.value()));
    }
    if (operation_attributes.output_mem_config.has_value()) {
        const auto& om = operation_attributes.output_mem_config.value();
        TT_FATAL(
            om.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "all_gather_regime_a_matmul_async v1 only supports an INTERLEAVED output memory config");
    }
    // NOTE: compute_kernel_config is hashed (so distinct configs never alias in cache) but the compute kernel is
    // always compiled with HiFi2 + fp32 dest accumulation (matching regime_a); a caller-supplied override is not
    // honored in v1. This is documented on the public op rather than rejected (it cannot corrupt output).
    // Not-yet-supported optional args: reject rather than silently ignore.
    TT_FATAL(
        !tensor_args.persistent_output_buffer.has_value(),
        "all_gather_regime_a_matmul_async v1 does not support persistent_output_buffer yet");
    TT_FATAL(
        !operation_attributes.barrier_semaphore.has_value(),
        "all_gather_regime_a_matmul_async v1 does not use barrier_semaphore");
    TT_FATAL(
        operation_attributes.num_links == 1 && operation_attributes.num_workers_per_link == 1,
        "all_gather_regime_a_matmul_async v1 uses one fabric link and one injector worker (got links={}, "
        "workers={})",
        operation_attributes.num_links,
        operation_attributes.num_workers_per_link);
    // Need >= 2*D global semaphores (D shard_ready + D shard_landed, per-shard local-first readiness).
    TT_FATAL(
        operation_attributes.multi_device_global_semaphore.size() >= 2u * operation_attributes.d,
        "all_gather_regime_a_matmul_async needs >= 2*D global semaphores, got {} for D={}",
        operation_attributes.multi_device_global_semaphore.size(),
        operation_attributes.d);
}

tt::stl::hash::hash_t AllGatherRegimeAMatmulAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& op, const tensor_args_t& tensor_args) {
    // Hash every field that changes codegen or geometry. GlobalSemaphore addresses are intentionally excluded
    // (they are relocated on cache replay by override_runtime_arguments, not structural). Semaphore COUNT (=D+1)
    // is covered by op.d. The full_gather_diagnostic flag is included so the two reader variants never alias.
    const auto& a = tensor_args.input_tensor.logical_shape();
    const auto& w = tensor_args.weight_tensor.logical_shape();
    const auto& cfg = op.regime_a_config;
    return tt::tt_metal::operation::hash_operation<AllGatherRegimeAMatmulAsyncDeviceOperation>(
        op.d,
        op.cluster_axis.has_value(),
        op.cluster_axis.value_or(0),
        static_cast<uint32_t>(op.topology),
        op.num_links,
        op.num_workers_per_link,
        op.num_buffers_per_channel,
        op.transport_c,
        op.transport_slots,
        op.packet_bytes,
        op.transport_mode,
        cfg.has_value(),
        cfg.has_value() ? cfg->k_slices : 0u,
        cfg.has_value() ? cfg->n_slices : 0u,
        cfg.has_value() ? cfg->m_slices : 0u,
        cfg.has_value() ? cfg->k_block_tiles : 0u,
        cfg.has_value() ? cfg->n_subblock_tiles : 0u,
        static_cast<uint32_t>(op.output_dtype.value_or(DataType::BFLOAT16)),
        op.output_mem_config.value_or(MemoryConfig{}),
        op.compute_kernel_config,
        static_cast<uint32_t>(tensor_args.input_tensor.dtype()),
        tensor_args.input_tensor.memory_config(),
        tensor_args.weight_tensor.memory_config(),
        a[-2],
        a[-1],
        w[-2],
        w[-1]);
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
            .compute_kernel_config = ckc,
            // Transport mode captured here so it is part of the hashed attributes (distinct program per mode).
            .transport_mode =
                [] {
                    const char* t = std::getenv("TT_AGMM_TRANSPORT");
                    if (t == nullptr) {
                        return 0u;  // default: ring_stream
                    }
                    if (std::string(t) == "source_to_all") {
                        return 1u;
                    }
                    if (std::string(t) == "full_wait") {
                        return 2u;
                    }
                    return 0u;
                }(),
            .multi_device_global_semaphore = std::move(multi_device_global_semaphore),
            .barrier_semaphore = std::move(barrier_semaphore)},
        tensor_args_t{
            .input_tensor = input_tensor,
            .weight_tensor = weight_tensor,
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
