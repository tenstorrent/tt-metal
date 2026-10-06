// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "selective_reduce_combine_device_operation.hpp"
#include "selective_reduce_combine_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

namespace detail {

// A row-major tensor has one-row pages when it is INTERLEAVED or HEIGHT_SHARDED (tt_metal page_config.cpp,
// get_page_shape_rm), DRAM or L1. WIDTH_SHARDED, BLOCK_SHARDED and ND sharding give (1, shard width) pages: a row then
// spans several pages and a writer's slice would have to be split at the shard boundaries, which neither writer does.
void validate_combine_output_memory_config(
    const tt::tt_metal::MemoryConfig& memory_config, uint32_t num_rows, uint32_t hidden_size, const char* what) {
    using tt::tt_metal::TensorMemoryLayout;
    const auto layout = memory_config.memory_layout();
    TT_FATAL(
        layout == TensorMemoryLayout::INTERLEAVED || layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "the [k, T, H] output is written one token row per TensorAccessor page and a row is not split across column "
        "pages, so the {} must be INTERLEAVED or HEIGHT_SHARDED row-major (WIDTH_SHARDED, BLOCK_SHARDED and ND "
        "sharding need a column-page split that is not implemented); got {}",
        what,
        memory_config);
    if (layout != TensorMemoryLayout::HEIGHT_SHARDED) {
        return;
    }
    const auto& shard_spec = memory_config.shard_spec();
    TT_FATAL(shard_spec.has_value(), "the {} is HEIGHT_SHARDED without a shard spec: {}", what, memory_config);
    TT_FATAL(
        shard_spec->shape[1] == hidden_size,
        "a height shard of the {} must hold whole token rows: shard width {} != hidden size {}",
        what,
        shard_spec->shape[1],
        hidden_size);
    const uint32_t shard_rows = shard_spec->shape[0];
    const uint32_t num_shard_cores = shard_spec->grid.num_cores();
    TT_FATAL(
        shard_rows > 0 && num_shard_cores * shard_rows >= num_rows,
        "the {} shard grid ({} cores x {} rows per shard) does not cover the k x T = {} token rows",
        what,
        num_shard_cores,
        shard_rows,
        num_rows);
}

}  // namespace detail

void SelectiveReduceCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.dense_input_tensor;
    const auto& token_activations_tensor = tensor_args.dense_activations_tensor;

    TT_FATAL(
        tensor_args.dense_token_maps_tensor.logical_shape().rank() == 2,
        "dense_token_maps_tensor must be rank 2 ([experts, per-token-stride]); got rank {}",
        tensor_args.dense_token_maps_tensor.logical_shape().rank());

    // Local combine mode: no fabric/links/mux, so skip link-related validation.
    if (!operation_attributes.local_combine) {
        const auto num_links = operation_attributes.num_links;
        TT_FATAL(num_links > 0, "num_links must be > 0, got {}", num_links);

        const auto worker_layout = detail::compute_worker_layout(
            input_tensor,
            operation_attributes.hidden_size,
            operation_attributes.num_token_parallel_cores,
            operation_attributes.num_data_parallel_cores);
        const auto num_worker_cores = worker_layout.num_worker_cores;
        TT_FATAL(
            num_worker_cores % num_links == 0,
            "num_worker_cores ({}) must be divisible by num_links ({})",
            num_worker_cores,
            num_links);
    }

    const auto batch_size = operation_attributes.batch_size;
    const auto seq_size = operation_attributes.seq_size;
    const auto total_tokens = batch_size * seq_size;

    // physical experts per device, replicated shared experts are counted per device (matches program factory)
    const uint32_t experts_per_device = tensor_args.dense_token_maps_tensor.logical_shape()[0];

    const uint32_t activations_stride_elm = token_activations_tensor.logical_shape()[-1] / total_tokens;

    const auto datum_size =
        tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(token_activations_tensor.dtype()));

    const auto alignment = (token_activations_tensor.memory_config().buffer_type() == BufferType::L1)
                               ? tt::tt_metal::hal::get_l1_alignment()
                               : tt::tt_metal::hal::get_dram_alignment();
    const uint32_t expected_activations_stride_elm =
        tt::align((2 * experts_per_device + 1) * datum_size, alignment) / datum_size;

    TT_FATAL(
        activations_stride_elm == expected_activations_stride_elm,
        "The token activations tensor is expected to have aligned 2 * experts_per_device + 1 elements per token");

    // dense_token_maps rows are (total_tokens + 1) token indices (the extra slot is the -1 terminator), each index
    // padded for NoC DMA. The kernels walk the maps CB as (e * (total_tokens + 1) + t) * dense_token_maps_stride_elm,
    // with the stride derived on the host as last_dim / (total_tokens + 1). Two things must hold for that indexing to
    // land on the right expert page, otherwise one expert's page silently aliases another's (or the walk runs off the
    // end of the CB): the row must divide evenly into (total_tokens + 1) indices, and the row must need no further
    // padding to reach the L1 page alignment the reader gives each expert page.
    const auto& dense_token_maps_tensor = tensor_args.dense_token_maps_tensor;
    const auto maps_row_elms = dense_token_maps_tensor.logical_shape()[-1];
    const auto maps_datum_size =
        tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(dense_token_maps_tensor.dtype()));

    TT_FATAL(
        maps_row_elms % (total_tokens + 1) == 0,
        "The dense token maps tensor is expected to hold (total_tokens + 1) = {} equally strided token indices per "
        "expert, but its last dim ({}) is not a multiple of that",
        total_tokens + 1,
        maps_row_elms);

    const auto maps_row_bytes = maps_row_elms * maps_datum_size;
    TT_FATAL(
        tt::align(maps_row_bytes, tt::tt_metal::hal::get_l1_alignment()) == maps_row_bytes,
        "The dense token maps tensor row ({} bytes) must already be L1 aligned ({} bytes); otherwise the per-expert "
        "page padding breaks the kernels' maps indexing",
        maps_row_bytes,
        tt::tt_metal::hal::get_l1_alignment());

    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    const auto& output_shape = output_spec.logical_shape();
    const uint32_t num_output_rows = output_shape[0] * output_shape[1];
    detail::validate_combine_output_memory_config(
        operation_attributes.output_memory_config, num_output_rows, output_shape[2], "output memory config");
    if (tensor_args.optional_output_tensor.has_value()) {
        detail::validate_combine_output_memory_config(
            tensor_args.optional_output_tensor->memory_config(),
            num_output_rows,
            output_shape[2],
            "optional output tensor memory config");
    }
}

SelectiveReduceCombineDeviceOperation::spec_return_value_t SelectiveReduceCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.dense_input_tensor;
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto hidden_size = operation_attributes.hidden_size;

    const uint32_t batch_size = operation_attributes.batch_size;
    const uint32_t seq_size = operation_attributes.seq_size;
    const uint32_t select_experts_k = operation_attributes.select_experts_k;

    const auto axis = operation_attributes.axis;
    const auto num_devices_cluster = (axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    const uint32_t total_tokens_per_device = batch_size * seq_size / num_devices_cluster;
    auto output_shape = ttnn::Shape({select_experts_k, total_tokens_per_device, hidden_size});

    auto mem_config = operation_attributes.output_memory_config;
    return tt::tt_metal::TensorSpec(
        Shape(output_shape), TensorLayout(input_tensor.dtype(), PageConfig(Layout::ROW_MAJOR), mem_config));
}

SelectiveReduceCombineDeviceOperation::tensor_return_value_t
SelectiveReduceCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_spec, tensor_args.dense_input_tensor.device()));
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::Tensor selective_reduce_combine(
    const ttnn::Tensor& dense_input_tensor,
    const ttnn::Tensor& dense_activations_tensor,
    const ttnn::Tensor& dense_token_maps_tensor,
    const ttnn::Tensor& dense_token_counts_tensor,
    uint32_t hidden_size,
    uint32_t batch_size,
    uint32_t seq_size,
    uint32_t select_experts_k,
    uint32_t cluster_axis,
    tt::tt_fabric::Topology topology,
    uint32_t num_links,
    uint32_t num_token_parallel_cores,
    uint32_t num_data_parallel_cores,
    const std::vector<ttnn::CoreCoord>& worker_cores,
    const CoreRangeSet& mux_core_range_set,
    const std::optional<ttnn::MemoryConfig>& output_memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore) {
    using OperationType = ttnn::experimental::prim::SelectiveReduceCombineDeviceOperation;
    auto memory_config = output_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .hidden_size = hidden_size,
            .batch_size = batch_size,
            .seq_size = seq_size,
            .select_experts_k = select_experts_k,
            .num_links = num_links,
            .axis = cluster_axis,
            .topology = topology,
            .num_token_parallel_cores = num_token_parallel_cores,
            .num_data_parallel_cores = num_data_parallel_cores,
            .worker_cores = worker_cores,
            .mux_core_range_set = mux_core_range_set,
            .output_memory_config = memory_config,
            .optional_cross_device_semaphore = optional_cross_device_semaphore},
        OperationType::tensor_args_t{
            .dense_input_tensor = dense_input_tensor,
            .dense_activations_tensor = dense_activations_tensor,
            .dense_token_maps_tensor = dense_token_maps_tensor,
            .dense_token_counts_tensor = dense_token_counts_tensor,
            .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
