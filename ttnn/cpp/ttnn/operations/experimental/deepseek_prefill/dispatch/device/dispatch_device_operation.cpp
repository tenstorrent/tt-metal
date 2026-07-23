// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "dispatch_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

// per_token_cast_to_fp8 block-wise quantization: 128 elements share one fp32 scale.
constexpr uint32_t numbers_per_scale_block = 128;

void DispatchDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.indices_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Indices tensor must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.expert_offsets_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Expert offsets tensor must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.expert_dispatch_table_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Expert dispatch table tensor must be ROW_MAJOR layout");

    // Validate input tensor dtypes. The input may be BFLOAT16 or FP8_E4M3: the tile path can
    // convert any input dtype to the output dtype via the compute packer, while the row-major path
    // is a pure byte copy (see the dtype-match check below).
    TT_FATAL(
        tensor_args.input_tensor.dtype() == DataType::BFLOAT16 ||
            tensor_args.input_tensor.dtype() == DataType::FP8_E4M3,
        "Input tensor must be BFLOAT16 or FP8_E4M3, got {}",
        tensor_args.input_tensor.dtype());
    TT_FATAL(
        tensor_args.indices_tensor.dtype() == DataType::UINT16,
        "Indices tensor must be UINT16 (matching moe_grouped_topk output), got {}",
        tensor_args.indices_tensor.dtype());
    TT_FATAL(
        tensor_args.expert_offsets_tensor.dtype() == DataType::INT32 ||
            tensor_args.expert_offsets_tensor.dtype() == DataType::UINT32,
        "Expert offsets tensor must be INT32 or UINT32, got {}",
        tensor_args.expert_offsets_tensor.dtype());
    TT_FATAL(
        tensor_args.expert_dispatch_table_tensor.dtype() == DataType::INT32,
        "Expert dispatch table tensor must be INT32, got {}",
        tensor_args.expert_dispatch_table_tensor.dtype());

    // FP8 (input or output) is Blackhole-only.
    if (operation_attributes.fp8_output) {
        TT_FATAL(
            tensor_args.input_tensor.device()->arch() != tt::ARCH::WORMHOLE_B0,
            "FP8 dispatch is not supported on Wormhole_B0; use Blackhole or set fp8_output=False");
    }

    // Row-major dispatch is a pure byte copy (no compute kernel), so it cannot convert dtypes:
    // the input dtype must equal the output dtype. fp8 output therefore requires fp8 input, and
    // bf16 output requires bf16 input. The tile path has a compute packer and converts freely.
    if (tensor_args.input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        const DataType output_dtype = operation_attributes.fp8_output ? DataType::FP8_E4M3 : DataType::BFLOAT16;
        TT_FATAL(
            tensor_args.input_tensor.dtype() == output_dtype,
            "Row-major dispatch is a byte copy: input dtype ({}) must match output dtype ({}). "
            "Use fp8 input for fp8 output, bf16 input for bf16 output, or TILE layout to convert.",
            tensor_args.input_tensor.dtype(),
            output_dtype);
    }

    // Validate output memory config is DRAM interleaved (not sharded)
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(),
        "Output memory config must be DRAM interleaved, not sharded");

    // Optional padding_config: per-device [local_real_tokens, pad_side], read on device to bound the
    // dispatch token loop. Must be ROW_MAJOR uint32/int32 with last dim 2.
    if (tensor_args.padding_config.has_value()) {
        const auto& padding_config = tensor_args.padding_config.value();
        TT_FATAL(
            padding_config.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "padding_config tensor must be ROW_MAJOR layout");
        TT_FATAL(
            padding_config.dtype() == DataType::UINT32 || padding_config.dtype() == DataType::INT32,
            "padding_config tensor must be UINT32 or INT32, got {}",
            padding_config.dtype());
        TT_FATAL(
            padding_config.logical_shape()[-1] == 2,
            "padding_config last dim must be 2 ([local_real_tokens, pad_side]), got {}",
            padding_config.logical_shape()[-1]);
    }

    // fp8-scaled-input path: the input is fp8 and each token carries its per-128-block (numbers_per_scale_block) scales
    // (ROW_MAJOR, last dim emb_dim/numbers_per_scale_block). Those scales are copied into the metadata tail
    // (fields 3..metadata_len-1), so metadata_len must reserve exactly those fields. Only valid on
    // the fp8 row-major (byte-copy) path. The flag and the scales tensor must be supplied together.
    if (operation_attributes.fp8_scaled_input) {
        TT_FATAL(
            tensor_args.scales_tensor.has_value(),
            "fp8_scaled_input requires a scales_tensor (per_token_cast_to_fp8 scales)");
        const auto& scales = tensor_args.scales_tensor.value();
        // scales layout is ROW_MAJOR
        TT_FATAL(scales.layout() == tt::tt_metal::Layout::ROW_MAJOR, "scales tensor must be ROW_MAJOR layout");
        // scales dtype is fp32 (TBD if other dtypes will be supported)
        TT_FATAL(scales.dtype() == DataType::FLOAT32, "scales tensor must be FLOAT32, got {}", scales.dtype());
        // scales is on the same device as the input
        TT_FATAL(
            scales.device() == tensor_args.input_tensor.device(),
            "scales tensor must be on the same device as the input tensor");
        // input is fp8 (FP8_E4M3) ROW_MAJOR
        TT_FATAL(
            tensor_args.input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
                tensor_args.input_tensor.dtype() == DataType::FP8_E4M3,
            "fp8_scaled_input requires a fp8 (FP8_E4M3) ROW_MAJOR input (the per_token_cast_to_fp8 "
            "compression path)");
        const uint32_t num_scales = scales.logical_shape()[-1];
        // metadata_len reserves 3 routing fields + one word per scale
        TT_FATAL(
            operation_attributes.metadata_len == 3 + num_scales,
            "metadata_len ({}) must equal 3 routing fields + scales last dim ({}) = {}",
            operation_attributes.metadata_len,
            num_scales,
            3 + num_scales);
        // total scale rows match total input tokens (flattened over all leading dims, as the reader does)
        const uint32_t input_hidden = tensor_args.input_tensor.logical_shape()[-1];
        const uint32_t input_tokens = tensor_args.input_tensor.logical_shape().volume() / input_hidden;
        const uint32_t scale_rows = scales.logical_shape().volume() / num_scales;
        TT_FATAL(
            scale_rows == input_tokens,
            "scales row count ({}) must match input token count ({})",
            scale_rows,
            input_tokens);
        // input hidden divisible by the scale block size with one scale per block
        TT_FATAL(
            input_hidden % numbers_per_scale_block == 0 && num_scales == input_hidden / numbers_per_scale_block,
            "fp8_scaled_input requires input hidden dim ({}) divisible by {} with one fp32 scale per "
            "block; got {} scale words (expected {})",
            input_hidden,
            numbers_per_scale_block,
            num_scales,
            input_hidden / numbers_per_scale_block);
    } else {
        TT_FATAL(
            !tensor_args.scales_tensor.has_value(),
            "scales_tensor was provided but fp8_scaled_input is false; pass fp8_scaled_input=True to use scales");
    }
}

void DispatchDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Empty for now
}

DispatchDeviceOperation::spec_return_value_t DispatchDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Extract necessary dimensions from operation attributes
    uint32_t metadata_len = operation_attributes.metadata_len;
    uint32_t max_dispatch_buffer_token_size = operation_attributes.max_dispatch_buffer_token_size;

    // Get the input tensor's per-device shape (sharded dimension)
    auto input_shape = tensor_args.input_tensor.tensor_spec().logical_shape();
    uint32_t hidden_dim = input_shape[-1];

    // Memory config for all output tensors (inherits sharding from input)
    auto mem_config = operation_attributes.output_mem_config;

    // Layout for all output tensors
    auto layout = tt::tt_metal::Layout::ROW_MAJOR;

    // Define output shapes - these are PER-DEVICE shapes (not global shapes). The
    // dispatch buffer is a single flat region shared across all local experts; its
    // total token capacity is max_dispatch_buffer_token_size.
    auto dispatch_buffer_shape = ttnn::Shape({1, 1, max_dispatch_buffer_token_size, hidden_dim});
    auto dispatch_metadata_shape = ttnn::Shape({1, 1, max_dispatch_buffer_token_size, metadata_len});

    // FP8 dispatch emits Fp8_e4m3 (1 byte/element); DataType::FP8_E4M3 maps directly to
    // tt::DataFormat::Fp8_e4m3 via datatype_to_dataformat_converter, so downstream CBs created
    // with detail::create_tensor_cb(output_tensor, ...) pick up the right dtype/page-size.
    auto dispatch_buffer_dtype = operation_attributes.fp8_output ? DataType::FP8_E4M3 : DataType::BFLOAT16;

    // Create tt::tt_metal::TensorSpec objects with correct dtypes
    auto dispatch_buffer_spec = tt::tt_metal::TensorSpec(
        Shape(dispatch_buffer_shape),
        tt::tt_metal::TensorLayout(dispatch_buffer_dtype, tt::tt_metal::PageConfig(layout), mem_config));

    auto dispatch_metadata_spec = tt::tt_metal::TensorSpec(
        Shape(dispatch_metadata_shape),
        tt::tt_metal::TensorLayout(DataType::INT32, tt::tt_metal::PageConfig(layout), mem_config));

    return {dispatch_buffer_spec, dispatch_metadata_spec};
}

DispatchDeviceOperation::topology_return_value_t DispatchDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Output tensors should have the same distribution topology as input tensor (sharded on dim 0)
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();

    // Both output tensors use explicit Shard placements across both mesh dimensions
    auto output_topology = tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(),
        {tt::tt_metal::distributed::MeshMapperConfig::Shard{0}, tt::tt_metal::distributed::MeshMapperConfig::Shard{1}},
        input_topology.mesh_coords());

    return {output_topology, output_topology};
}

DispatchDeviceOperation::tensor_return_value_t DispatchDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto output_tensor = create_device_tensor(output_spec[0], tensor_args.input_tensor.device());
    auto metadata_tensor = create_device_tensor(output_spec[1], tensor_args.input_tensor.device());
    return {output_tensor, metadata_tensor};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch

namespace ttnn::prim {
ttnn::operations::experimental::deepseek_prefill::dispatch::DispatchDeviceOperation::tensor_return_value_t
prefill_dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatch_buffer_token_size,
    const std::optional<ttnn::Tensor>& padding_config,
    const std::optional<ttnn::Tensor>& scales_tensor,
    std::optional<uint32_t> axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set,
    bool use_l1_small_for_semaphores,
    bool fp8_output,
    bool fp8_scaled_input,
    uint32_t num_workers_per_sender) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::dispatch::DispatchDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dispatch_group_size = dispatch_group_size,
            .experts_per_chip = experts_per_chip,
            .num_routed_experts = num_routed_experts,
            .num_experts_per_tok = num_experts_per_tok,
            .metadata_len = metadata_len,
            .max_dispatch_buffer_token_size = max_dispatch_buffer_token_size,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .output_mem_config = memory_config,
            .worker_core_range_set = worker_core_range_set,
            .use_l1_small_for_semaphores = use_l1_small_for_semaphores,
            .fp8_output = fp8_output,
            .num_workers_per_sender = num_workers_per_sender,
            .has_padding_config = padding_config.has_value(),
            .fp8_scaled_input = fp8_scaled_input},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .indices_tensor = indices_tensor,
            .expert_offsets_tensor = expert_offsets_tensor,
            .expert_dispatch_table_tensor = expert_dispatch_table_tensor,
            .padding_config = padding_config,
            .scales_tensor = scales_tensor});
}
}  // namespace ttnn::prim
