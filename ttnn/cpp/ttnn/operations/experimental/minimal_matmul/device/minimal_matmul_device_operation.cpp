// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_device_operation.hpp"
#include "minimal_matmul_program_factory.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

MinimalMatmulDeviceOperation::program_factory_t MinimalMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MinimalMatmulProgramFactory{};
}

void MinimalMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const bool has_bias = tensor_args.bias_tensor.has_value();
    const Tensor* bias_ptr = has_bias ? &tensor_args.bias_tensor.value() : nullptr;
    const auto& config = operation_attributes.config;

    // Basic device/storage checks
    TT_FATAL(
        act_tensor.storage_type() == StorageType::DEVICE && weight_tensor.storage_type() == StorageType::DEVICE,
        "minimal_matmul operands must be on device");
    TT_FATAL(act_tensor.device() == weight_tensor.device(), "minimal_matmul inputs must reside on the same device");
    TT_FATAL(
        act_tensor.buffer() != nullptr && weight_tensor.buffer() != nullptr,
        "minimal_matmul inputs must be allocated in device buffers");
    if (has_bias) {
        const auto& bias_tensor = *bias_ptr;
        TT_FATAL(bias_tensor.storage_type() == StorageType::DEVICE, "minimal_matmul bias must be on device");
        TT_FATAL(bias_tensor.device() == act_tensor.device(), "minimal_matmul bias must be on the same device");
        TT_FATAL(bias_tensor.buffer() != nullptr, "minimal_matmul bias must be allocated in a device buffer");
    }

    // Layout requirements: all inputs must be TILE layout
    TT_FATAL(
        act_tensor.layout() == Layout::TILE && weight_tensor.layout() == Layout::TILE,
        "minimal_matmul requires TILE layout for activation and weight");
    if (has_bias) {
        TT_FATAL(bias_ptr->layout() == Layout::TILE, "minimal_matmul requires TILE layout for bias");
    }

    // DType constraints: support BFLOAT16, BFLOAT8_B, BFLOAT4_B and FLOAT32
    auto dtype_supported = [](tt::tt_metal::DataType dt) {
        return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B ||
               dt == DataType::FLOAT32;
    };
    TT_FATAL(
        dtype_supported(act_tensor.dtype()) && dtype_supported(weight_tensor.dtype()),
        "minimal_matmul supports only BFLOAT16, BFLOAT8_B, BFLOAT4_B, and FLOAT32 for inputs");

    // Bias dtype constraint, if present
    if (has_bias) {
        TT_FATAL(
            dtype_supported(bias_ptr->dtype()),
            "minimal_matmul supports only BFLOAT16, BFLOAT8_B, and BFLOAT4_B for bias");
    }

    // Shape constraints
    const auto& a_logical = act_tensor.logical_shape();
    const auto& w_logical = weight_tensor.logical_shape();
    TT_FATAL(a_logical.rank() >= 2 && w_logical.rank() >= 2, "minimal_matmul expects rank >= 2 tensors");

    // Allow upper-dim broadcasting on activation (LHS): activation may have arbitrary upper dims
    for (int i = 0; i < static_cast<int>(w_logical.rank()) - 2; ++i) {
        TT_FATAL(w_logical[i] == 1, "minimal_matmul weight must have 1 in all dims < -2");
    }

    const uint32_t M = a_logical[-2];
    const uint32_t K = a_logical[-1];
    const uint32_t K_w = w_logical[-2];
    const uint32_t N = w_logical[-1];

    TT_FATAL(K == K_w, "minimal_matmul inner dimensions must match, got K={} and K_w={}", K, K_w);
    TT_FATAL(M > 0 && K > 0 && N > 0, "minimal_matmul dimensions must be positive");

    // Validate chunks and dim parameters
    const int32_t chunks = operation_attributes.chunks;
    const int32_t dim = operation_attributes.dim;
    TT_FATAL(chunks >= 1, "minimal_matmul requires chunks >= 1, got chunks={}", chunks);
    TT_FATAL(dim == -1, "minimal_matmul currently only supports dim=-1, got dim={}", dim);

    if (chunks > 1) {
        // Validate N is divisible by chunks
        TT_FATAL(N % chunks == 0, "Output width N={} must be divisible by chunks={}", N, chunks);

        // Validate each chunk is tile-aligned
        const uint32_t N_per_chunk = N / chunks;
        TT_FATAL(
            N_per_chunk % tt::constants::TILE_WIDTH == 0,
            "Each chunk size N/chunks={} must be a multiple of TILE_WIDTH={}",
            N_per_chunk,
            tt::constants::TILE_WIDTH);
    }

    if (has_bias) {
        const auto& b_logical = bias_ptr->logical_shape();
        TT_FATAL(b_logical.rank() >= 1, "minimal_matmul bias must have rank >= 1");
        // All dims except the last must be 1 (i.e., shape is [..., 1, N])
        for (int i = 0; i < static_cast<int>(b_logical.rank()) - 1; ++i) {
            TT_FATAL(b_logical[i] == 1, "minimal_matmul bias must be 1 in all dims except the last");
        }
        TT_FATAL(b_logical[-1] == N, "minimal_matmul bias last dimension must equal N ({}), got {}", N, b_logical[-1]);
    }

    // Tile alignment checks (implicitly guaranteed by TILE layout, but assert inner two dims are tile-aligned)
    const auto& a_padded = act_tensor.padded_shape();
    const auto& w_padded = weight_tensor.padded_shape();
    TT_FATAL(
        a_padded[-2] % TILE_HEIGHT == 0 && a_padded[-1] % TILE_WIDTH == 0,
        "minimal_matmul activation must be tile-aligned");
    TT_FATAL(
        w_padded[-2] % TILE_HEIGHT == 0 && w_padded[-1] % TILE_WIDTH == 0,
        "minimal_matmul weight must be tile-aligned");
    if (has_bias) {
        const auto& b_padded = bias_ptr->padded_shape();
        TT_FATAL(b_padded[-1] % TILE_WIDTH == 0, "minimal_matmul bias last dimension must be tile-aligned");
    }

    // Validate fused ternary tensors if present
    bool has_fused_ternary = operation_attributes.fused_ternary_scalar.has_value();
    if (has_fused_ternary) {
        TT_FATAL(
            tensor_args.fused_ternary_input_a.has_value() && tensor_args.fused_ternary_input_b.has_value(),
            "If fused_ternary_scalar is provided, both fused_ternary_input_a and fused_ternary_input_b must be "
            "provided");

        TT_FATAL(
            !operation_attributes.fused_activation.has_value(),
            "minimal_matmul does not support using fused_activation together with ternary inputs "
            "(dit_minimal_matmul_addcmul_fused). "
            "Please use either fused_activation or ternary inputs, not both.");

        const auto& ternary_a = tensor_args.fused_ternary_input_a.value();
        const auto& ternary_b = tensor_args.fused_ternary_input_b.value();

        TT_FATAL(ternary_a.storage_type() == StorageType::DEVICE, "fused_ternary_input_a must be on device");
        TT_FATAL(ternary_b.storage_type() == StorageType::DEVICE, "fused_ternary_input_b must be on device");
        TT_FATAL(ternary_a.device() == act_tensor.device(), "fused_ternary_input_a must be on same device");
        TT_FATAL(ternary_b.device() == act_tensor.device(), "fused_ternary_input_b must be on same device");
        TT_FATAL(ternary_a.buffer() != nullptr, "fused_ternary_input_a must be allocated");
        TT_FATAL(ternary_b.buffer() != nullptr, "fused_ternary_input_b must be allocated");

        TT_FATAL(ternary_a.layout() == Layout::TILE, "fused_ternary_input_a must be TILE layout");
        TT_FATAL(ternary_b.layout() == Layout::TILE, "fused_ternary_input_b must be TILE layout");

        TT_FATAL(
            dtype_supported(ternary_a.dtype()) && dtype_supported(ternary_b.dtype()),
            "fused_ternary tensors must have supported dtypes");

        const auto& ternary_a_logical = ternary_a.logical_shape();
        const auto& ternary_b_logical = ternary_b.logical_shape();

        // ternary_a matches output [M, N], ternary_b is broadcast [1, N]
        TT_FATAL(
            ternary_a_logical[-2] == M && ternary_a_logical[-1] == N,
            "fused_ternary_input_a shape must match output [M={}, N={}], got [{}, {}]",
            M,
            N,
            ternary_a_logical[-2],
            ternary_a_logical[-1]);
        TT_FATAL(
            ternary_b_logical[-2] == 1 && ternary_b_logical[-1] == N,
            "fused_ternary_input_b shape must be [1, N={}] (broadcast like bias), got [{}, {}]",
            N,
            ternary_b_logical[-2],
            ternary_b_logical[-1]);
    }

    // Config constraints
    if (config.has_value()) {
        const auto& cfg = config.value();
        TT_FATAL(cfg.M_block_size > 0 && cfg.K_block_size > 0 && cfg.N_block_size > 0, "Block sizes must be > 0");
        TT_FATAL(cfg.subblock_h > 0 && cfg.subblock_w > 0, "Subblock sizes must be > 0");
        TT_FATAL(
            (cfg.M_block_size % cfg.subblock_h) == 0,
            "M_block_size ({}) must be divisible by subblock_h ({})",
            cfg.M_block_size,
            cfg.subblock_h);
        TT_FATAL(
            (cfg.N_block_size % cfg.subblock_w) == 0,
            "N_block_size ({}) must be divisible by subblock_w ({})",
            cfg.N_block_size,
            cfg.subblock_w);

        // Grid must be at least 1x1
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
            "compute_with_storage_grid_size must be >= 2x2");

        // Additional grid checks are performed when creating the program
        auto device_grid = act_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x <= device_grid.x &&
                cfg.compute_with_storage_grid_size.y <= device_grid.y,
            "compute_with_storage_grid_size must be <= device grid size");

        const uint32_t max_dest_volume = get_dest_reg_count(operation_attributes.compute_kernel_config);
        TT_FATAL(
            cfg.subblock_h * cfg.subblock_w <= max_dest_volume, "subblock_h * subblock_w must be <= max_dest_volume");
    }
}

MinimalMatmulDeviceOperation::spec_return_value_t MinimalMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& in0_input_tensor = tensor_args.input_tensor;
    const auto& in1_input_tensor = tensor_args.weight_tensor;
    const auto& in0_input_tensor_shape = in0_input_tensor.logical_shape();
    const auto& in1_input_tensor_shape = in1_input_tensor.logical_shape();
    const uint32_t N = in1_input_tensor_shape[-1];
    const int32_t chunks = operation_attributes.chunks;

    const auto& memory_config = operation_attributes.output_mem_config.value_or(in0_input_tensor.memory_config());
    auto dtype = operation_attributes.output_dtype.value_or(in0_input_tensor.dtype());

    // Create specs for output tensors
    std::vector<TensorSpec> output_specs;
    output_specs.reserve(chunks);

    const uint32_t N_per_chunk = N / chunks;
    for (int32_t i = 0; i < chunks; ++i) {
        ttnn::Shape output_shape(in0_input_tensor_shape);
        output_shape[-1] = N_per_chunk;
        output_specs.push_back(TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config)));
    }

    return output_specs;
}

MinimalMatmulDeviceOperation::tensor_return_value_t MinimalMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input_tensor.device();

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_specs.size());

    for (const auto& spec : output_specs) {
        output_tensors.push_back(create_device_tensor(spec, device));
    }

    return output_tensors;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> minimal_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<Tensor>& bias_tensor,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    int32_t chunks,
    int32_t dim,
    std::optional<float> fused_ternary_scalar,
    const std::optional<Tensor>& fused_ternary_input_a,
    const std::optional<Tensor>& fused_ternary_input_b) {
    using OperationType = experimental::prim::MinimalMatmulDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .fused_activation = std::move(fused_activation),
            .output_mem_config = memory_config,
            .output_dtype = dtype,
            .fused_ternary_scalar = fused_ternary_scalar,
            .compute_kernel_config = kernel_config_val,
            .chunks = chunks,
            .dim = dim},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .weight_tensor = weight_tensor,
            .bias_tensor = bias_tensor,
            .optional_input_tensor = std::nullopt,
            .fused_ternary_input_a = fused_ternary_input_a,
            .fused_ternary_input_b = fused_ternary_input_b});
}

}  // namespace ttnn::prim
