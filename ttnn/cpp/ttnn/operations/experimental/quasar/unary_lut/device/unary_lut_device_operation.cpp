// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_lut_device_operation.hpp"

#include <tt-metalium/hal_types.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::quasar::unary_lut {

ttsl::hash::hash_t UnaryLutDeviceOperation::operation_attributes_t::to_hash() const {
    // Fold the LUT config (eval method, degrees, segment count, coefficient data) into
    // the hash so different activations get distinct JIT-compiled kernels and program
    // cache entries — the LUT is baked into the kernel via -D defines.
    std::size_t lut_sig = 0;
    if (lut_config.has_value()) {
        const auto& l = *lut_config;
        lut_sig = ttsl::hash::hash_objects_with_default_seed(
            l.eval_method,
            l.poly_degree,
            l.num_segments,
            l.num_degree,
            l.den_degree,
            l.data,
            l.rr_method,
            l.rr_log_ln2,
            l.rr_exp_mult,
            l.rr_exp_const,
            l.rr_scale0,
            l.rr_scale1,
            l.rr_scale2,
            l.rr_exp2_mult,
            l.rr_compose,
            l.rr_log2_scale,
            l.rr_log2_basis_mminus1,
            l.rr_input_offset,
            l.rr_pow_n,
            l.rr_pow_recip,
            l.nr_magic,
            l.nr_c1,
            l.nr_c2,
            l.nr_iters,
            l.nr_n,
            l.nr_reciprocal);
    }
    return ttsl::hash::hash_objects_with_default_seed(memory_config, input_dtype, lut_sig);
}

void UnaryLutDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "unary_lut: input tensor must be on device, got storage type: {}",
        input.storage_type());
    TT_FATAL(input.layout() == Layout::TILE, "unary_lut: input must be TILE layout, got {}", input.layout());

    // This op exists to prove the unary-LUT DFB path on the smallest config:
    // fully height/block-sharded L1, bf16, TILE 32x32.
    TT_FATAL(input.is_sharded(), "unary_lut: input must be sharded (fully-sharded L1 DFB slice)");
    const auto& mc = input.memory_config();
    TT_FATAL(
        mc.buffer_type() == BufferType::L1 && (mc.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
                                               mc.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED),
        "unary_lut: input must be height/block-sharded L1");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16, "unary_lut: only bf16 supported in this DFB slice, got {}", input.dtype());

    const auto& tile = input.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "unary_lut: only 32x32 tiles supported");

    TT_FATAL(attributes.memory_config.is_sharded(), "unary_lut: output memory config must be sharded");

    if (tensor_args.output_tensor.has_value()) {
        const auto& out = *tensor_args.output_tensor;
        TT_FATAL(
            out.logical_shape() == input.logical_shape(),
            "unary_lut: output shape {} must match input shape {}",
            out.logical_shape(),
            input.logical_shape());
        TT_FATAL(out.is_sharded(), "unary_lut: output tensor must be sharded");
    }
}

void UnaryLutDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

UnaryLutDeviceOperation::spec_return_value_t UnaryLutDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor->tensor_spec();
    }
    const auto& input = tensor_args.input_tensor;
    // Same shape / layout / dtype as the input; output reuses the (sharded) memory config.
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(attributes.input_dtype, PageConfig(Layout::TILE), attributes.memory_config));
}

UnaryLutDeviceOperation::tensor_return_value_t UnaryLutDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t UnaryLutDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    TT_FATAL(is_device_tensor(input), "unary_lut: unexpected tensor type {}", input.storage_type());
    // The DFB ProgramSpec (shard grid, per-core tile counts, DFB sizing) depends on the
    // shard volume / total tile count, so fold the logical shape into the hash.
    return operation::hash_operation<UnaryLutDeviceOperation>(
        attributes, input.dtype(), input.memory_config(), input.logical_shape().volume());
}

bool UnaryLutDeviceOperation::skip_launch(
    const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t& output) {
    return output.logical_shape().volume() == 0;
}

UnaryLutDeviceOperation::program_factory_t UnaryLutDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactoryMetalV2{};
}

}  // namespace ttnn::operations::experimental::quasar::unary_lut

namespace ttnn::prim::qsr {

ttnn::operations::experimental::quasar::unary_lut::UnaryLutDeviceOperation::tensor_return_value_t unary_lut(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<ttnn::operations::experimental::quasar::unary_lut::LutConfig>& lut_config) {
    using OperationType = ttnn::operations::experimental::quasar::unary_lut::UnaryLutDeviceOperation;

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "unary_lut: input tensor must be on device, got storage type: {}",
        input_tensor.storage_type());

    MemoryConfig mem_config_actual = memory_config.value_or(
        optional_output_tensor.has_value() ? optional_output_tensor->memory_config() : input_tensor.memory_config());

    // Worker grid = the input shard grid (native L1 sharding).
    auto* device = input_tensor.device();
    CoreRangeSet worker_grid;
    if (sub_device_id.has_value()) {
        worker_grid = device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
    } else {
        const auto& shard_grid = input_tensor.shard_spec()->grid;
        worker_grid = shard_grid;
        for (const auto& sd_id : device->get_sub_device_ids()) {
            const auto& sd_workers = device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
            if (sd_workers.intersects(shard_grid)) {
                worker_grid = sd_workers.intersection(shard_grid);
                break;
            }
        }
    }

    auto operation_attributes = OperationType::operation_attributes_t{
        mem_config_actual,
        input_tensor.dtype(),
        worker_grid,
        sub_device_id,
        lut_config,
    };
    auto tensor_args = OperationType::tensor_args_t{input_tensor, optional_output_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim::qsr
