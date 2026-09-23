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

    const auto& query = tensor_args.query.logical_shape();
    const auto& key_cache = tensor_args.key_cache.logical_shape();
    const auto& weights = tensor_args.head_weights.logical_shape();
    TT_FATAL(
        query[-1] == key_cache[-1],
        "fused_lightning_select_kv: query D {} must match key_cache D {}",
        query[-1],
        key_cache[-1]);
    TT_FATAL(key_cache[1] == 1, "fused_lightning_select_kv: key_cache heads must be 1, got {}", key_cache[1]);
    TT_FATAL(
        query[1] == weights[-1] && query[-2] == weights[-2],
        "fused_lightning_select_kv: head_weights must be [B, 1, Sq, Hi] matching query heads {} and Sq {}, got {}",
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
    TT_FATAL(
        query[0] == pt_shape[0],
        "fused_lightning_select_kv: query batch {} must match page_table batch {}",
        query[0],
        pt_shape[0]);

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
    const uint32_t batch = tensor_args.page_table_tensor.logical_shape()[0];
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({batch, kv_shape[1], args.k, kv_shape[3]}),
        tt::tt_metal::TensorLayout(kv.dtype(), tt::tt_metal::PageConfig(kv.layout()), args.output_mem_config));
}

FusedLightningSelectKvDeviceOperation::tensor_return_value_t
FusedLightningSelectKvDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.kv_cache.device());
}

tt::tt_metal::ProgramDescriptor FusedLightningSelectKvDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    TT_THROW("fused_lightning_select_kv: device kernel is not implemented");
}

}  // namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv

namespace ttnn::prim {

ttnn::Tensor fused_lightning_select_kv(
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
