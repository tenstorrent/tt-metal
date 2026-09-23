// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_lightning_select_kv_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv {

using namespace tt::tt_metal;

namespace {

void require_rank4_device(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "fused_lightning_select_kv: {} must be on device", name);
    TT_FATAL(tensor.logical_shape().rank() == 4, "fused_lightning_select_kv: {} must be rank-4", name);
}

}  // namespace

FusedLightningSelectKvDeviceOperation::program_factory_t FusedLightningSelectKvDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void FusedLightningSelectKvDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    require_rank4_device(tensor_args.query, "query");
    require_rank4_device(tensor_args.key_cache, "key_cache");
    require_rank4_device(tensor_args.head_weights, "head_weights");
    require_rank4_device(tensor_args.kv_cache, "kv_cache");

    TT_FATAL(args.k > 0, "fused_lightning_select_kv: k must be positive, got {}", args.k);
    TT_FATAL(
        tensor_args.kv_cache.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "fused_lightning_select_kv: kv_cache must be row-major");
    const auto& query = tensor_args.query.logical_shape();
    const auto& key_cache = tensor_args.key_cache.logical_shape();
    const auto& kv_cache = tensor_args.kv_cache.logical_shape();

    log_info(tt::LogOp, "query: {}, key_cache: {}, kv_cache: {}", query, key_cache, kv_cache);
    TT_FATAL(
        key_cache[-2] == kv_cache[-2],
        "fused_lightning_select_kv: key_cache and kv_cache must have the same page block size, got {} and {}",
        key_cache[-2],
        kv_cache[-2]);
    const auto& weights = tensor_args.head_weights.logical_shape();
    TT_FATAL(
        query[-1] == key_cache[-1],
        "fused_lightning_select_kv: query D {} must match key_cache D {}",
        query[-1],
        key_cache[-1]);
    TT_FATAL(key_cache[1] == 1, "fused_lightning_select_kv: key_cache heads must be 1, got {}", key_cache[1]);
    TT_FATAL(
        key_cache[-1] % (2 * tt::constants::TILE_WIDTH) == 0,
        "fused_lightning_select_kv: D {} must be a multiple of {} (custom_mm needs an even number of K tiles)",
        key_cache[-1],
        2 * tt::constants::TILE_WIDTH);
    TT_FATAL(
        key_cache[-2] % tt::constants::TILE_HEIGHT == 0,
        "fused_lightning_select_kv: page block size {} must be a multiple of {}",
        key_cache[-2],
        tt::constants::TILE_HEIGHT);
    {
        const auto& key_mem = tensor_args.key_cache.memory_config();
        const auto& key_padded = tensor_args.key_cache.padded_shape();
        TT_FATAL(
            tensor_args.key_cache.layout() == tt::tt_metal::Layout::TILE && key_mem.is_dram() &&
                key_mem.nd_shard_spec().has_value(),
            "fused_lightning_select_kv: key_cache must be a tiled DRAM ND-sharded tensor");
        const auto& shard_shape = key_mem.nd_shard_spec()->shard_shape;
        TT_FATAL(
            shard_shape.volume() == key_padded[-2] * key_padded[-1] && shard_shape[-2] == key_padded[-2] &&
                shard_shape[-1] == key_padded[-1],
            "fused_lightning_select_kv: key_cache shard shape must be one block [1, 1, {}, {}], got {}",
            key_padded[-2],
            key_padded[-1],
            shard_shape);
    }
    TT_FATAL(
        query[1] == weights[-1] && query[-2] == weights[-2],
        "fused_lightning_select_kv: head_weights must be [num_cores, 1, 1, Hi] matching query heads {} and Sq {}, got "
        "{}",
        query[1],
        query[-2],
        weights);
    TT_FATAL(weights[1] == 1, "fused_lightning_select_kv: head_weights dim 1 must be 1, got {}", weights[1]);

    const auto& page_table = tensor_args.page_table_tensor;
    const auto& cur_pos = tensor_args.cur_pos_tensor;
    TT_FATAL(
        page_table.storage_type() == StorageType::DEVICE,
        "fused_lightning_select_kv: page_table_tensor must be on device");
    TT_FATAL(
        cur_pos.storage_type() == StorageType::DEVICE, "fused_lightning_select_kv: cur_pos_tensor must be on device");
    TT_FATAL(page_table.dtype() == DataType::INT32, "fused_lightning_select_kv: page_table_tensor must be INT32");
    TT_FATAL(cur_pos.dtype() == DataType::INT32, "fused_lightning_select_kv: cur_pos_tensor must be INT32");

    const auto& pt_shape = page_table.logical_shape();
    const auto& pos_shape = cur_pos.logical_shape();
    TT_FATAL(
        pt_shape.rank() == 2,
        "fused_lightning_select_kv: page_table_tensor must be [B, max_blocks_per_user], got {}",
        pt_shape);
    TT_FATAL(
        pos_shape.rank() == 1 && pos_shape[0] == pt_shape[0],
        "fused_lightning_select_kv: cur_pos_tensor must be [B] with B = {}, got {}",
        pt_shape[0],
        pos_shape);

    // Decode only: one user (B == 1) and one query token (Sq == 1).
    const uint32_t batch = pt_shape[0];
    TT_FATAL(batch == 1, "fused_lightning_select_kv: only B == 1 is supported, got page_table batch {}", batch);
    TT_FATAL(query[2] == 1, "fused_lightning_select_kv: only Sq == 1 is supported, got query Sq {}", query[2]);
    TT_FATAL(
        key_cache[0] == kv_cache[0],
        "fused_lightning_select_kv: key_cache and kv_cache must have the same number of blocks, got {} and {}",
        key_cache[0],
        kv_cache[0]);

    // query is a ROW_MAJOR HEIGHT_SHARDED L1 tensor with one full [Hi, D] replica per core, stacked
    // along the batch axis, so its logical shape is [num_cores, Hi, 1, D].
    const auto& query_tensor = tensor_args.query;
    TT_FATAL(
        query_tensor.layout() == Layout::ROW_MAJOR,
        "fused_lightning_select_kv: query must be ROW_MAJOR, got {}",
        query_tensor.layout());
    TT_FATAL(
        query_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "fused_lightning_select_kv: query must be HEIGHT_SHARDED, got {}",
        query_tensor.memory_config().memory_layout());
    TT_FATAL(
        query_tensor.buffer()->buffer_type() == BufferType::L1, "fused_lightning_select_kv: query must be L1-resident");
    TT_FATAL(
        query_tensor.memory_config().shard_spec().has_value(),
        "fused_lightning_select_kv: query must carry a shard spec");
    const auto& query_shard = query_tensor.memory_config().shard_spec().value();
    TT_FATAL(
        query_shard.orientation == ShardOrientation::ROW_MAJOR,
        "fused_lightning_select_kv: query requires ROW_MAJOR shard orientation");
    TT_FATAL(
        query_shard.shape[0] == query[1] && query_shard.shape[1] == query[-1],
        "fused_lightning_select_kv: query shard must be one full [Hi, D] = [{}, {}] replica per core, got [{}, {}]",
        query[1],
        query[-1],
        query_shard.shape[0],
        query_shard.shape[1]);
    TT_FATAL(
        query[-1] % tt::constants::TILE_WIDTH == 0,
        "fused_lightning_select_kv: query D {} must be a multiple of the 1x32 tile width",
        query[-1]);
    TT_FATAL(
        query[1] % 8 == 0,
        "fused_lightning_select_kv: query Hi {} must be a multiple of 8 (the query is tilized into 8x32 tiles)",
        query[1]);
    const uint32_t num_query_cores = query_shard.grid.num_cores();
    TT_FATAL(
        query[0] == num_query_cores,
        "fused_lightning_select_kv: query batch {} must equal the shard grid's core count {} (one replica per core)",
        query[0],
        num_query_cores);

    // head_weights uses the same layout: one full [1, Hi] replica per core on the query's grid, so
    // its logical shape is [num_cores, 1, 1, Hi].
    const auto& weights_tensor = tensor_args.head_weights;
    TT_FATAL(
        weights_tensor.layout() == Layout::ROW_MAJOR,
        "fused_lightning_select_kv: head_weights must be ROW_MAJOR, got {}",
        weights_tensor.layout());
    TT_FATAL(
        weights_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "fused_lightning_select_kv: head_weights must be HEIGHT_SHARDED, got {}",
        weights_tensor.memory_config().memory_layout());
    TT_FATAL(
        weights_tensor.buffer()->buffer_type() == BufferType::L1,
        "fused_lightning_select_kv: head_weights must be L1-resident");
    TT_FATAL(
        weights_tensor.memory_config().shard_spec().has_value(),
        "fused_lightning_select_kv: head_weights must carry a shard spec");
    const auto& weights_shard = weights_tensor.memory_config().shard_spec().value();
    TT_FATAL(
        weights_shard.orientation == ShardOrientation::ROW_MAJOR,
        "fused_lightning_select_kv: head_weights requires ROW_MAJOR shard orientation");
    TT_FATAL(
        weights_shard.grid == query_shard.grid,
        "fused_lightning_select_kv: head_weights shard grid {} must equal the query shard grid {}",
        weights_shard.grid.str(),
        query_shard.grid.str());
    TT_FATAL(
        weights_shard.shape[0] == 1 && weights_shard.shape[1] == weights[-1],
        "fused_lightning_select_kv: head_weights shard must be one full [1, Hi] = [1, {}] replica per core, got "
        "[{}, {}]",
        weights[-1],
        weights_shard.shape[0],
        weights_shard.shape[1]);
    TT_FATAL(
        weights[-1] % tt::constants::TILE_WIDTH == 0,
        "fused_lightning_select_kv: head_weights Hi {} must be a multiple of the 1x32 tile width",
        weights[-1]);
    TT_FATAL(
        weights[0] == num_query_cores,
        "fused_lightning_select_kv: head_weights batch {} must equal the shard grid's core count {} (one replica "
        "per core)",
        weights[0],
        num_query_cores);

    if (tensor_args.valid_length_tensor.has_value()) {
        TT_FATAL(
            tensor_args.valid_length_tensor->storage_type() == StorageType::DEVICE,
            "fused_lightning_select_kv: valid_length_tensor must be on device");
    }
}

FusedLightningSelectKvDeviceOperation::spec_return_value_t FusedLightningSelectKvDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& kv = tensor_args.kv_cache;
    const auto& kv_shape = kv.logical_shape();
    const auto& pt_shape = tensor_args.page_table_tensor.logical_shape();
    const uint32_t batch = pt_shape[0];
    const uint32_t max_keys = pt_shape[1] * tensor_args.key_cache.logical_shape()[-2];
    return {
        tt::tt_metal::TensorSpec(
            ttnn::Shape({batch, kv_shape[1], args.k, kv_shape[3]}),
            tt::tt_metal::TensorLayout(kv.dtype(), tt::tt_metal::PageConfig(kv.layout()), args.output_mem_config)),
        // One fp32 score per key the page table can address; only the first valid ones are written.
        tt::tt_metal::TensorSpec(
            ttnn::Shape({batch, 1, 1, max_keys}),
            tt::tt_metal::TensorLayout(
                DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), tt::tt_metal::MemoryConfig{})),
    };
}

FusedLightningSelectKvDeviceOperation::tensor_return_value_t
FusedLightningSelectKvDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, tensor_args.kv_cache.device()));
    }
    return outputs;
}

}  // namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv

namespace ttnn::prim {

std::vector<ttnn::Tensor> fused_lightning_select_kv(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key_cache,
    const ttnn::Tensor& head_weights,
    const ttnn::Tensor& kv_cache,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& cur_pos_tensor,
    uint32_t k,
    const std::optional<ttnn::Tensor>& valid_length_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::experimental::deepseek::fused_lightning_select_kv::FusedLightningSelectKvDeviceOperation;

    TT_FATAL(kv_cache.storage_type() == StorageType::DEVICE, "fused_lightning_select_kv: kv_cache must be on device");

    auto kernel_config = init_device_compute_kernel_config(
        kv_cache.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);

    auto attrs = OperationType::operation_attributes_t{
        .k = k,
        .output_mem_config = memory_config.value_or(tt::tt_metal::MemoryConfig{}),
        .compute_kernel_config = kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .query = query,
        .key_cache = key_cache,
        .head_weights = head_weights,
        .kv_cache = kv_cache,
        .page_table_tensor = page_table_tensor,
        .cur_pos_tensor = cur_pos_tensor,
        .valid_length_tensor = valid_length_tensor,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
