// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_up_matmul_decode_device_operation.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tt-metalium/work_split.hpp"

namespace ttnn::operations::matmul_decode {

GateUpMatmulDecodeDeviceOperation::program_factory_t GateUpMatmulDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return GateUpPartialWidthSharded{};
}

void GateUpMatmulDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& gate_b = tensor_args.gate_b;
    const auto& up_b = tensor_args.up_b;

    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Input tensor A must be in tile layout");
    TT_FATAL(gate_b.layout() == Layout::TILE, "gate_b must be in tile layout");
    TT_FATAL(up_b.layout() == Layout::TILE, "up_b must be in tile layout");

    // gate_b and up_b must share geometry (same K/N split, same core grid) so the single
    // gather + the dual-weight compute kernel can drive both with one set of dims.
    TT_FATAL(
        gate_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            up_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "gate_b and up_b must both be width-sharded (partial-width-sharded weights)");
    const auto& gss = gate_b.memory_config().shard_spec();
    const auto& uss = up_b.memory_config().shard_spec();
    TT_FATAL(gss.has_value() && uss.has_value(), "gate_b and up_b must have shard specs");
    TT_FATAL(
        gss->shape[0] == uss->shape[0] && gss->shape[1] == uss->shape[1],
        "gate_b shard shape [{}, {}] must equal up_b shard shape [{}, {}]",
        gss->shape[0],
        gss->shape[1],
        uss->shape[0],
        uss->shape[1]);
    TT_FATAL(
        gss->grid == uss->grid, "gate_b and up_b must be sharded over the SAME core grid (shared gather geometry)");

    // reshard_input is mandatory for this fused op (it mirrors the production gate/up calls,
    // which always pass reshard_input=True).
    TT_FATAL(operation_attributes.reshard_input, "gate_up_matmul_decode requires reshard_input=True");
    const auto in0_layout = input_tensor_a.memory_config().memory_layout();
    TT_FATAL(
        in0_layout == TensorMemoryLayout::INTERLEAVED || in0_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            in0_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "reshard_input requires Input tensor A to be interleaved, block-sharded, or width-sharded, but got {}",
        in0_layout);
    const uint32_t reshard_cores = operation_attributes.reshard_cores;
    TT_FATAL(reshard_cores > 0, "reshard_cores must be > 0");
    TT_FATAL(
        operation_attributes.K % reshard_cores == 0,
        "reshard_input requires K ({}) to be divisible by reshard_cores ({})",
        operation_attributes.K,
        reshard_cores);
    TT_FATAL(
        (operation_attributes.K / reshard_cores) % tt::constants::TILE_WIDTH == 0,
        "reshard_input requires the per-core K-slice (K/reshard_cores = {}) to be tile-aligned",
        operation_attributes.K / reshard_cores);
    TT_FATAL(
        input_tensor_a.logical_shape()[-1] == operation_attributes.K,
        "Input tensor A must have the same K dimension as the operation attributes");
    TT_FATAL(
        input_tensor_a.logical_shape()[-2] == operation_attributes.M,
        "Input tensor A must have the same M dimension as the operation attributes");
}

GateUpMatmulDecodeDeviceOperation::spec_return_value_t GateUpMatmulDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;

    ttnn::Shape output_shape(input_tensor_a.logical_shape());
    output_shape[-1] = operation_attributes.N;

    const auto dtype = operation_attributes.output_dtype.value_or(input_tensor_a.dtype());

    // The single GeGLU output is width-sharded across N_blocks cores with shard [M, Nc] --
    // identical geometry to a single partial-width-sharded matmul_decode; the downstream
    // down-projection reshards it as its K.
    int per_core_output_width = tt::constants::TILE_WIDTH;
    const auto& b_mem = tensor_args.gate_b.memory_config();
    if (b_mem.is_sharded() && b_mem.shard_spec().has_value()) {
        per_core_output_width = b_mem.shard_spec().value().shape[1];
    }
    int output_num_cores = tt::div_up(operation_attributes.N, per_core_output_width);
    CoreRangeSet output_core_range_set = tt::tt_metal::num_cores_to_corerangeset(
        output_num_cores, input_tensor_a.device()->compute_with_storage_grid_size(), true);
    std::array<uint32_t, 2> shard_shape = {
        static_cast<uint32_t>(operation_attributes.M), static_cast<uint32_t>(per_core_output_width)};
    auto shard_spec =
        tt::tt_metal::ShardSpec(output_core_range_set, shard_shape, tt::tt_metal::ShardOrientation::ROW_MAJOR);
    auto memory_config = MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec);

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, input_tensor_a.tensor_spec().tile()),
            memory_config));
}

GateUpMatmulDecodeDeviceOperation::tensor_return_value_t GateUpMatmulDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input_tensor_a.device());
}

}  // namespace ttnn::operations::matmul_decode

namespace ttnn::prim {
ttnn::operations::matmul_decode::GateUpMatmulDecodeDeviceOperation::tensor_return_value_t gate_up_matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& gate_b,
    const Tensor& up_b,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    bool fused_gelu_approx,
    bool reshard_input,
    uint32_t reshard_cores) {
    using OperationType = ttnn::operations::matmul_decode::GateUpMatmulDecodeDeviceOperation;

    // gate_b/up_b are reshaped/permuted partial-width-sharded weights, so recover N (per-output
    // width) from the shard geometry + A's K (mirrors matmul_decode's partial_width_sharded path).
    const int M = input_tensor_a.logical_shape()[-2];
    const int K_a = input_tensor_a.logical_shape()[-1];
    const int K_b = gate_b.logical_shape()[-2];
    int N = gate_b.logical_shape()[-1];
    if (K_a >= K_b) {
        TT_FATAL(K_a % K_b == 0, "K_a must be divisible by K_b");
        const int K_ratio = K_a / K_b;
        N = N / K_ratio;
    }
    const int K = K_a;
    log_debug(tt::LogOp, "gate_up_matmul_decode with M={}, N={}, K={}", M, N, K);

    auto operation_attributes = OperationType::operation_attributes_t{
        M,
        N,
        K,
        input_tensor_a.memory_config(),
        dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
        compute_kernel_config,
        fused_gelu_approx,
        reshard_input,
        reshard_cores,
    };
    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, gate_b, up_b};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
