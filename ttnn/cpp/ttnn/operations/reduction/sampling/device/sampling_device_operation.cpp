// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/sampling/device/sampling_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <optional>

#include "ttnn/operations/reduction/sampling/device/sampling_device_operation_types.hpp"
#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include <tt-metalium/tt_backend_api_types.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {
void SamplingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_values_tensor = tensor_args.input_values;
    const auto& input_indices_tensor = tensor_args.input_indices;
    const auto& k = tensor_args.k;
    const auto& p = tensor_args.p;
    const auto& temp = tensor_args.temp;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    // WH/BH support both UINT32 and INT32 for the index/k/output dtypes. Every other architecture
    // (e.g. Quasar, which lacks UInt16/UInt32 tile (DFB) metadata support) runs the kernels in the
    // 32-bit (INT32) path, so UINT32 is not representable and must be rejected instead of silently
    // miscomputing. Gated on !(WH || BH) so new architectures default to the safe 32-bit path.
    const auto arch = input_values_tensor.device()->arch();
    const bool use_32bit_index = !(arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE);

    TT_FATAL(
        input_values_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(input_values_tensor.dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_values_tensor.layout() == Layout::TILE, "Only TILE_LAYOUT is supported for inputs!");

    TT_FATAL(
        input_indices_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(
        input_indices_tensor.dtype() == DataType::UINT32 || input_indices_tensor.dtype() == DataType::INT32,
        "Only UINT32 & INT32 dtypes are supported for input indices!");
    TT_FATAL(
        !use_32bit_index || input_indices_tensor.dtype() == DataType::INT32,
        "Only INT32 is supported for input indices on this architecture (UINT32 is only available on "
        "Wormhole/Blackhole)!");

    TT_FATAL(input_indices_tensor.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR is supported for input indices!");

    TT_FATAL(
        input_indices_tensor.logical_shape() == input_values_tensor.logical_shape(),
        "Input values and indices must have the same shape!");
    auto input_shape = input_values_tensor.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "Sampling input_values must be rank-4; got rank {}", input_shape.rank());

    // Users live in dim 2 (H): the output is sized [1,1,1,H] and the per-user inputs/cores are
    // indexed by the user (row) index. dims 0 and 1 (N, C) must therefore be 1; otherwise
    // num_users = N*C*H would exceed H and the writer would index past the H-sized output.
    TT_FATAL(
        input_shape[0] == 1 && input_shape[1] == 1,
        "Sampling requires input dims 0 and 1 (N, C) to be 1; got [{}, {}, {}, {}]!",
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3]);
    // N and C are guaranteed to be 1 by the check above, so the user count is just dim 2 (H).
    const uint32_t num_users = input_shape[2];
    TT_FATAL(
        num_users >= 1 && num_users <= 32,
        "Sampling currently supports between 1 and 32 users (one core per user); got {}!",
        num_users);
    TT_FATAL(
        input_shape[3] != 0 && input_shape[3] % 32 == 0,
        "Input inner dim ({}) must be non-zero and divisible by 32, pad if needed!",
        input_shape[3]);
    // The top-k stage processes the W/32 candidate tiles with a pairwise local sort followed by a
    // bitonic merge tree whose schedule assumes a power-of-2 tile count. A non-power-of-2 Wt hangs
    // the device (odd Wt) or silently drops candidate tiles (even non-power-of-2 Wt), so reject it
    // here instead of timing out. See https://github.com/tenstorrent/tt-metal/issues/44558.
    const uint32_t Wt = input_shape[3] / 32;
    TT_FATAL(
        (Wt & (Wt - 1)) == 0,
        "Input inner dim ({}) must yield a power-of-2 number of tiles (Wt = W/32 = {}); pad W up to the "
        "next power-of-2 multiple of 32 (e.g. with -inf values and dummy indices) if needed!",
        input_shape[3],
        Wt);

    if (args.sub_core_grids.has_value()) {
        ReduceOpDeviceGridValidationOptions sampling_grid_opts;
        sampling_grid_opts.num_cores_use_last_core_divider = true;
        sampling_grid_opts.sub_grid_contained_in_device_grid = &args.sub_core_grids.value();
        sampling_grid_opts.sub_grid_label = "sub_core_grids";
        validate_reduce_op_tensor(input_values_tensor, "Sampling", "input_values", &sampling_grid_opts);
    }
    if (args.sub_core_grids.has_value()) {
        // The grid may be over-provisioned: only the first `num_users` cores are used, any extras
        // are ignored. It must supply at least `num_users` cores (one per user).
        TT_FATAL(
            args.sub_core_grids.value().num_cores() >= num_users,
            "Subcore grid must supply at least num_users ({}) cores, but found {}!",
            num_users,
            args.sub_core_grids.value().num_cores());
    }
    if (preallocated_output_tensor.has_value()) {
        TT_FATAL(
            preallocated_output_tensor.value().dtype() == DataType::UINT32 ||
                preallocated_output_tensor.value().dtype() == DataType::INT32,
            "Only UINT32 & INT32 dtypes are supported for outputs!");
        TT_FATAL(
            !use_32bit_index || preallocated_output_tensor.value().dtype() == DataType::INT32,
            "Only INT32 is supported for outputs on this architecture (UINT32 is only available on "
            "Wormhole/Blackhole)!");

        TT_FATAL(
            preallocated_output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");

        const auto& sampling_pre_out = preallocated_output_tensor.value();
        const auto& sampling_pre_out_shape = sampling_pre_out.logical_shape();
        TT_FATAL(
            sampling_pre_out_shape.rank() == 4,
            "Sampling preallocated output must be rank-4, got rank {}",
            sampling_pre_out_shape.rank());
        TT_FATAL(
            sampling_pre_out_shape[0] == 1 && sampling_pre_out_shape[1] == 1 && sampling_pre_out_shape[2] == 1 &&
                sampling_pre_out_shape[3] == input_shape[2],
            "Sampling preallocated output logical shape must be [1,1,1,{}] (input dim 2), got [{},{},{},{}]",
            input_shape[2],
            sampling_pre_out_shape[0],
            sampling_pre_out_shape[1],
            sampling_pre_out_shape[2],
            sampling_pre_out_shape[3]);
    }

    // Check size, layout and dtype of k, p, temp
    TT_FATAL(
        k.dtype() == DataType::UINT32 || k.dtype() == DataType::INT32,
        "Only UINT32 & INT32 dtypes are supported for k!");
    TT_FATAL(
        !use_32bit_index || k.dtype() == DataType::INT32,
        "Only INT32 is supported for k on this architecture (UINT32 is only available on "
        "Wormhole/Blackhole)!");
    TT_FATAL(p.dtype() == DataType::BFLOAT16, "Only BFLOAT16 dtypes are supported for p!");
    TT_FATAL(temp.dtype() == DataType::BFLOAT16, "Only BFLOAT16 dtypes are supported for temp!");
    TT_FATAL(k.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for k!");
    TT_FATAL(p.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for p!");
    TT_FATAL(temp.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for temp!");
    // k/p/temp carry one entry per user, so they must match num_users (== N*C*H). Only num_users
    // cores run and each reads its own entry, so no padding to 32 is required.
    TT_FATAL(k.logical_shape() == Shape({num_users}), "k must have shape [{}] (one per user)!", num_users);
    TT_FATAL(p.logical_shape() == Shape({num_users}), "p must have shape [{}] (one per user)!", num_users);
    TT_FATAL(temp.logical_shape() == Shape({num_users}), "temp must have shape [{}] (one per user)!", num_users);
}

tt::tt_metal::TensorSpec SamplingDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_values_tensor = tensor_args.input_values;
    auto input_shape = input_values_tensor.logical_shape();
    ttnn::Shape output_shape({1, 1, 1, input_shape[2]});

    // WH/BH keep the historical UINT32 default; every other architecture (e.g. Quasar) runs the
    // 32-bit path and cannot represent a UInt32 tile, so the default (non-preallocated) output must
    // be INT32 there. Gated on !(WH || BH) so new architectures default to the safe INT32 path.
    const auto arch = input_values_tensor.device()->arch();
    const bool use_32bit_index = !(arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE);
    const DataType output_dtype = use_32bit_index ? DataType::INT32 : DataType::UINT32;

    return tt::tt_metal::TensorSpec(
        output_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), input_values_tensor.memory_config()));
}

Tensor SamplingDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_values.device());
}

ttnn::Tensor sampling(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& k,
    const Tensor& p,
    const Tensor& temp,
    const std::optional<uint32_t>& seed,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const std::optional<Tensor>& preallocated_output_tensor) {
    return ttnn::device_operation::launch<SamplingDeviceOperation>(
        SamplingParams{.seed = seed, .sub_core_grids = sub_core_grids},
        SamplingInputs{
            .input_values = input_values_tensor,
            .input_indices = input_indices_tensor,
            .k = k,
            .p = p,
            .temp = temp,
            .preallocated_output = preallocated_output_tensor});
}

}  // namespace ttnn::prim
