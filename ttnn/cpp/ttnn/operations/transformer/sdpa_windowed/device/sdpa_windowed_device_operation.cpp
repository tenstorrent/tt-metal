// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa_windowed/device/sdpa_windowed_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/transformer/sdpa_windowed/device/sdpa_windowed_program_factory.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/constants.hpp>
#include <cmath>
#include <enchantum/enchantum.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

WindowedScaledDotProductAttentionDeviceOperation::program_factory_t
WindowedScaledDotProductAttentionDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return WindowedSDPAProgramFactory{};
}

void WindowedScaledDotProductAttentionDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    validate_on_program_cache_miss(attrs, tensors);
}

void WindowedScaledDotProductAttentionDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    // Common validations for windowed SDPA
    const auto& q = tensors.q;
    const auto& k = tensors.k;
    const auto& v = tensors.v;
    const auto& cu_window_seqlens = tensors.cu_window_seqlens;

    // Check cu_window_seqlens shape
    const auto& cu_window_seqlens_shape = cu_window_seqlens.logical_shape();
    TT_FATAL(cu_window_seqlens_shape.rank() == 1, "cu_window_seqlens must be a 1D tensor");
    TT_FATAL(cu_window_seqlens_shape[0] >= 2, "cu_window_seqlens must have at least 2 elements");
    TT_FATAL(
        cu_window_seqlens_shape[0] <= cu_window_seqlens_nelements,
        "cu_window_seqlens must have less than {} elements",
        cu_window_seqlens_nelements);
    TT_FATAL(
        cu_window_seqlens.dtype() == DataType::UINT32 || cu_window_seqlens.dtype() == DataType::INT32,
        "cu_window_seqlens must be uint32 or int32");

    // Check storage and dtype
    for (const auto& input_tensor : {&q, &k, &v}) {
        TT_FATAL(input_tensor->storage_type() == StorageType::DEVICE, "Operands to windowed SDPA need to be on device");
        TT_FATAL(
            input_tensor->buffer() != nullptr, "Operands to windowed SDPA need to be allocated in buffers on device");
        TT_FATAL(
            input_tensor->buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to windowed SDPA need to be in DRAM");
        TT_FATAL((input_tensor->layout() == Layout::TILE), "Inputs to windowed SDPA must be tilized");
        TT_FATAL(
            input_tensor->dtype() == DataType::BFLOAT16 || input_tensor->dtype() == DataType::BFLOAT8_B ||
                input_tensor->dtype() == DataType::BFLOAT4_B,
            "Data type of input tensor must be BFLOAT16, BFLOAT8_B, or BFLOAT4_B and is {}",
            input_tensor->dtype());
    }

    TT_FATAL(cu_window_seqlens.storage_type() == StorageType::DEVICE, "cu_window_seqlens must be on device");
    TT_FATAL(cu_window_seqlens.buffer() != nullptr, "cu_window_seqlens must be allocated in buffers on device");
    TT_FATAL(
        cu_window_seqlens.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "cu_window_seqlens must be in DRAM");
    TT_FATAL((cu_window_seqlens.layout() == Layout::ROW_MAJOR), "cu_window_seqlens must be in row-major layout");

    auto validate_padding = [&](const Tensor& tensor) {
        auto logical_shape = tensor.logical_shape();
        auto padded_shape = tensor.padded_shape();
        TT_FATAL(
            logical_shape[0] == padded_shape[0],
            "Padding is not supported on the batch dimension and is {}",
            logical_shape[0]);
        TT_FATAL(
            logical_shape[1] == padded_shape[1],
            "Padding is not supported on the num_heads dimension and is {}",
            logical_shape[1]);
        TT_FATAL(
            logical_shape[3] == padded_shape[3],
            "Padding is not supported on the head_dim dimension and is {}",
            logical_shape[3]);
    };

    auto validate_regular_mode = [&]() {
        // Shape checks
        const auto q_shape = q.logical_shape();
        const auto k_shape = k.logical_shape();
        const auto v_shape = v.logical_shape();
        const auto B = q_shape[0];
        const auto nqh = q_shape[1];
        const auto nkv = k_shape[1];
        const auto DH = q_shape[3];
        const auto Sk = k_shape[2];

        TT_FATAL(
            k_shape[0] == B && v_shape[0] == B, "K and V batch must match. Got K: {}, V: {}", k_shape[0], v_shape[0]);
        TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
        TT_FATAL(v_shape[2] == Sk, "K and V sequence length must match. Got K: {}, V: {}", k_shape[2], v_shape[2]);
        TT_FATAL(
            k_shape[3] == DH && v_shape[3] == DH,
            "K and V hidden dim must match. Got K: {}, V: {}",
            k_shape[3],
            v_shape[3]);
        TT_FATAL(
            nqh >= nkv && nqh % nkv == 0,
            "Q num_heads must be >= K num_heads and divisible by K num_heads. Got Q: {}, K: {}",
            nqh,
            nkv);

        if (attrs.program_config.has_value()) {
            auto q_chunk_size = attrs.program_config->q_chunk_size;
            auto k_chunk_size = attrs.program_config->k_chunk_size;

            TT_FATAL(
                q_chunk_size % tt::constants::TILE_WIDTH == 0,
                "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
                q_chunk_size,
                tt::constants::TILE_WIDTH);
            TT_FATAL(
                k_chunk_size % tt::constants::TILE_WIDTH == 0,
                "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
                k_chunk_size,
                tt::constants::TILE_WIDTH);
        }
    };

    validate_regular_mode();

    // Check padding: Only the sequence dimension may be padded. For all other dims, logical shape must be equal to
    // padded shape
    validate_padding(q);
    validate_padding(k);
    validate_padding(v);
}

TensorSpec WindowedScaledDotProductAttentionDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    const auto& input = tensors.q;
    return TensorSpec(
        input.logical_shape(), TensorLayout(input.dtype(), PageConfig(Layout::TILE), attrs.output_mem_config));
}

Tensor WindowedScaledDotProductAttentionDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(attrs, tensors), tensors.q.device());
}

tt::stl::hash::hash_t WindowedScaledDotProductAttentionDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    operation::Hash hash = operation::hash_operation<WindowedScaledDotProductAttentionDeviceOperation>(
        attrs.scale,
        attrs.output_mem_config,
        attrs.program_config,
        attrs.compute_kernel_config,
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.cu_window_seqlens);
    return hash;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor>
WindowedScaledDotProductAttentionDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& tensors, tensor_return_value_t& output_tensor) {
    Tensors input_tensors = {tensors.q, tensors.k, tensors.v, tensors.cu_window_seqlens};
    if (output_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    MathFidelity math_fidelity = ttnn::get_math_fidelity(attrs.compute_kernel_config);
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "Windowed SDPA perf model does not support tt::arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<tensor_return_value_t>(input_tensors, output_tensor, 0);
    }

    // For windowed attention, we only compute attention within windows
    // Calculate total number of attention computations based on windows
    int64_t num_mul_adds = 0;

    CoreCoord compute_grid_dims = output_tensor.device()->compute_with_storage_grid_size();
    int num_cores = compute_grid_dims.x * compute_grid_dims.y;

    const int tensix_mul_adds_per_cycle_lofi = 4096;

    int ideal_dev_clock_cycles = std::ceil(
        ((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        (float)operation::OpPerformanceModel::fidelity_multiplier(math_fidelity));

    return operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        input_tensors, output_tensor, ideal_dev_clock_cycles);
}

Tensor windowed_scaled_dot_product_attention(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& cu_window_seqlens,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = WindowedScaledDotProductAttentionDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .output_mem_config = output_mem_config,
            .program_config = std::move(program_config),
            .compute_kernel_config = compute_kernel_config,
        },
        OperationType::tensor_args_t{
            .q = input_tensor_q,
            .k = input_tensor_k,
            .v = input_tensor_v,
            .cu_window_seqlens = cu_window_seqlens,
        });
}

}  // namespace ttnn::prim
