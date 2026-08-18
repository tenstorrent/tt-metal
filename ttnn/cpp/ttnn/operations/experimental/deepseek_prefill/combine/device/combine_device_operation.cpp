// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "combine_device_operation.hpp"
#include "combine_sf.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine {

namespace {

// Slots per ring are not an operation parameter: the caller sizes the staging tensor and the op
// divides it evenly across every (direction, level, sender core) stream, so the buffer itself is
// the knob.  The exact page-size equality check lives in the program factory, which is the first
// place the output tensor -- and therefore the payload stride the page must accommodate -- exists.
void validate_store_and_forward(const CombineParams& attrs, const CombineInputs& tensors) {
    if (!attrs.use_store_and_forward) {
        TT_FATAL(
            !tensors.staging_buffer.has_value(),
            "staging_buffer must not be supplied when use_store_and_forward is false; passing one "
            "would let the program-cache key and the buffer list disagree");
        return;
    }

    auto* mesh_device = tensors.dispatched_buffer.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t axis = attrs.axis.value_or(0);
    const uint32_t extent = axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();
    const bool is_ring = attrs.topology == tt::tt_fabric::Topology::Ring;
    const uint32_t levels = sf::num_levels(extent, is_ring);

    if (levels == 0) {
        // No token on this mesh is more than one hop from its destination, so nothing is ever
        // staged and the flag is inert.  Accept it without a buffer so a caller can leave the flag
        // on across mesh shapes.
        TT_FATAL(
            !tensors.staging_buffer.has_value(),
            "use_store_and_forward needs no staging_buffer on a {}-device {} axis: max distance is "
            "{}, so no token is ever relayed",
            extent,
            is_ring ? "ring" : "linear",
            sf::max_distance(extent, is_ring));
        return;
    }

    TT_FATAL(
        tensors.staging_buffer.has_value(),
        "use_store_and_forward requires a staging_buffer on a {}-device {} axis ({} relay level(s))",
        extent,
        is_ring ? "ring" : "linear",
        levels);

    const auto& staging = tensors.staging_buffer.value();
    TT_FATAL(staging.device() == mesh_device, "staging_buffer must live on the same mesh device");
    TT_FATAL(
        staging.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "staging_buffer must be ROW_MAJOR so one page is exactly one staged token, got {}",
        staging.layout());
    TT_FATAL(
        staging.dtype() == DataType::UINT32,
        "staging_buffer must be UINT32 -- it is opaque scratch, and a fixed dtype keeps its page "
        "size independent of the combine output dtype -- got {}",
        staging.dtype());
    TT_FATAL(
        staging.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
            !staging.memory_config().is_sharded(),
        "staging_buffer must be interleaved DRAM");

    const uint32_t num_cores = std::min(attrs.num_links, 4u);
    const uint32_t streams = 2u * levels * num_cores;
    const uint32_t staging_pages = staging.buffer()->num_pages();
    TT_FATAL(
        staging_pages % streams == 0,
        "staging_buffer has {} pages, which does not divide evenly across {} streams "
        "(2 directions x {} level(s) x {} sender core(s))",
        staging_pages,
        streams,
        levels,
        num_cores);

    const uint32_t slots = staging_pages / streams;
    TT_FATAL(
        slots >= 2 && (slots & (slots - 1)) == 0,
        "staging_buffer yields {} slot(s) per stream; must be a power of two and at least 2 (the "
        "slot index is a mask, and a single slot cannot overlap a read with a write)",
        slots);
}

}  // namespace

void CombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate layouts
    TT_FATAL(
        tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::TILE ||
            tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Dispatched buffer must be TILE_LAYOUT or ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.dispatched_metadata.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Dispatched metadata must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.expert_token_counts.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Experts token counter must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.expert_region_offsets.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Expert region offsets must be ROW_MAJOR layout");

    // Validate dtypes
    TT_FATAL(
        tensor_args.dispatched_buffer.dtype() == DataType::BFLOAT16 ||
            (tensor_args.dispatched_buffer.dtype() == DataType::BFLOAT8_B &&
             tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::TILE),
        "Dispatched buffer must be BFLOAT16 or BFLOAT8_B with TILE layout, got {}",
        tensor_args.dispatched_buffer.dtype());
    TT_FATAL(
        tensor_args.dispatched_metadata.dtype() == DataType::INT32,
        "Dispatched metadata must be INT32, got {}",
        tensor_args.dispatched_metadata.dtype());
    TT_FATAL(
        tensor_args.expert_token_counts.dtype() == DataType::INT32 ||
            tensor_args.expert_token_counts.dtype() == DataType::UINT32,
        "Experts token counter must be INT32 or UINT32, got {}",
        tensor_args.expert_token_counts.dtype());
    TT_FATAL(
        tensor_args.expert_region_offsets.dtype() == DataType::INT32 ||
            tensor_args.expert_region_offsets.dtype() == DataType::UINT32,
        "Expert region offsets must be INT32 or UINT32, got {}",
        tensor_args.expert_region_offsets.dtype());
    TT_FATAL(
        tensor_args.expert_region_offsets.tensor_spec().logical_shape() ==
            tensor_args.expert_token_counts.tensor_spec().logical_shape(),
        "expert_region_offsets shape {} must match expert_token_counts shape {}",
        tensor_args.expert_region_offsets.tensor_spec().logical_shape(),
        tensor_args.expert_token_counts.tensor_spec().logical_shape());

    // Validate output memory config
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(),
        "Output memory config must be interleaved (L1 or DRAM), not sharded");

    // FP8 output is only supported on Blackhole, and only when the dispatched buffer is TILE.
    // The BF16 -> FP8 conversion happens in the packer when the untilize compute kernel writes
    // tiles into the untilize CB (which inherits the output tensor's Fp8_e4m3 DataFormat).
    // The ROW_MAJOR path has no untilize stage, so the reader would forward unconverted BF16
    // bytes through FP8-sized output pages, producing corrupt output (local writes) or
    // overrunning the output-sized writer CB (non-local writes).
    if (operation_attributes.use_fp8_combine) {
        auto arch = tensor_args.dispatched_buffer.device()->arch();
        TT_FATAL(arch == tt::ARCH::BLACKHOLE, "use_fp8_combine=true requires Blackhole hardware (got arch {})", arch);
        TT_FATAL(
            tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::TILE,
            "use_fp8_combine=true requires TILE-layout dispatched buffer; only the TILE path "
            "retargets the untilize CB to Fp8_e4m3 (got layout {})",
            tensor_args.dispatched_buffer.layout());
    }

    // Validate tensor shapes are compatible
    // Dispatch outputs are 4D: (per_device_batch, 1, max_dispatch_buffer_token_size, hidden_dim/metadata_len)
    // Counter is 3D: (num_dispatch_groups, per_device_batch, num_routed_experts)
    auto dispatched_shape = tensor_args.dispatched_buffer.tensor_spec().logical_shape();
    auto metadata_shape = tensor_args.dispatched_metadata.tensor_spec().logical_shape();
    auto counter_shape = tensor_args.expert_token_counts.tensor_spec().logical_shape();

    TT_FATAL(
        dispatched_shape[0] == metadata_shape[0] && dispatched_shape[0] == counter_shape[0],
        "First dimension (per_device_batch) must match across all input tensors");
    TT_FATAL(
        dispatched_shape[2] == metadata_shape[2],
        "Flat buffer dim must match: dispatched[2]={} vs metadata[2]={}",
        dispatched_shape[2],
        metadata_shape[2]);
    TT_FATAL(
        counter_shape[-1] % operation_attributes.experts_per_chip == 0,
        "counter last dim (num_routed_experts={}) must be divisible by experts_per_chip={}",
        counter_shape[-1],
        operation_attributes.experts_per_chip);

    validate_store_and_forward(operation_attributes, tensor_args);
}

void CombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Empty for now
}

CombineDeviceOperation::spec_return_value_t CombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Get input shape to extract hidden_dim
    auto dispatched_shape = tensor_args.dispatched_buffer.tensor_spec().logical_shape();
    uint32_t hidden_dim = dispatched_shape[-1];

    // Output shape: (1, 1, seq_len_per_chip, num_experts_per_tok, hidden_dim)
    auto output_shape = ttnn::Shape(
        {1, 1, operation_attributes.seq_len_per_chip, operation_attributes.num_experts_per_tok, hidden_dim});

    // Memory config and layout
    auto mem_config = operation_attributes.output_mem_config;
    auto layout = tt::tt_metal::Layout::ROW_MAJOR;

    // FP8 combine emits Fp8_e4m3 (1 byte/element); DataType::FP8_E4M3 maps directly to
    // tt::DataFormat::Fp8_e4m3 via datatype_to_dataformat_converter, so downstream CBs created
    // with detail::create_tensor_cb(output_tensor, ...) will pick up the right dtype/page-size.
    auto output_dtype = operation_attributes.use_fp8_combine ? DataType::FP8_E4M3 : DataType::BFLOAT16;

    auto output_spec = tt::tt_metal::TensorSpec(
        Shape(output_shape), tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(layout), mem_config));

    return output_spec;
}

CombineDeviceOperation::tensor_return_value_t CombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.dispatched_buffer.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine

namespace ttnn::prim {
ttnn::Tensor prefill_combine(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    std::optional<uint32_t> axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set,
    bool init_zeros,
    bool use_l1_small_for_semaphores,
    bool use_fp8_combine,
    const std::optional<ttnn::Tensor>& staging_buffer,
    bool use_store_and_forward) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::combine::CombineDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dispatch_group_size = dispatch_group_size,
            .experts_per_chip = experts_per_chip,
            .num_experts_per_tok = num_experts_per_tok,
            .seq_len_per_chip = seq_len_per_chip,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .output_mem_config = memory_config,
            .worker_core_range_set = worker_core_range_set,
            .init_zeros = init_zeros,
            .use_l1_small_for_semaphores = use_l1_small_for_semaphores,
            .use_fp8_combine = use_fp8_combine,
            .use_store_and_forward = use_store_and_forward},
        OperationType::tensor_args_t{
            .dispatched_buffer = dispatched_buffer,
            .dispatched_metadata = dispatched_metadata,
            .expert_token_counts = expert_token_counts,
            .expert_region_offsets = expert_region_offsets,
            .staging_buffer = staging_buffer});
}
}  // namespace ttnn::prim
