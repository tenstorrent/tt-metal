// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/attn_res_merge/device/attn_res_merge_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

void AttnResMergeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& partial = tensor_args.partial;
    const auto& prefix_sum = tensor_args.prefix_sum;

    // The full-width operands are BFLOAT16 only: the MAC path is
    // `mul_tiles_bcast_cols` with acc_to_dest, and that is the one numeric
    // configuration gated against a reference.
    //
    // The row scalars also take FLOAT32, because a caller whose score chain runs
    // in fp32 shouldn't have to spend a typecast to hand them over. They must
    // agree with each other: all three share one circular buffer, so the
    // derivation reads them through a single unpack configuration.
    operations::check_tensor(partial, "AttnResMerge", "partial", {DataType::BFLOAT16});
    operations::check_tensor(prefix_sum, "AttnResMerge", "prefix_sum", {DataType::BFLOAT16});

    const std::array<std::pair<const ttnn::Tensor*, const char*>, 3> scalars = {{
        {&tensor_args.shift, "shift"},
        {&tensor_args.mass, "mass"},
        {&tensor_args.live_scores, "live_scores"},
    }};
    for (const auto& [tensor, name] : scalars) {
        operations::check_tensor(*tensor, "AttnResMerge", name, {DataType::BFLOAT16, DataType::FLOAT32});
        TT_FATAL(
            tensor->dtype() == tensor_args.shift.dtype(),
            "AttnResMerge requires one dtype across shift, mass and live_scores; {} is {} against shift's {}",
            name,
            tensor->dtype(),
            tensor_args.shift.dtype());
    }

    const std::array<std::pair<const ttnn::Tensor*, const char*>, 5> operands = {{
        {&partial, "partial"},
        {&prefix_sum, "prefix_sum"},
        {&tensor_args.shift, "shift"},
        {&tensor_args.mass, "mass"},
        {&tensor_args.live_scores, "live_scores"},
    }};
    for (const auto& [tensor, name] : operands) {
        TT_FATAL(
            tensor->storage_type() == StorageType::DEVICE,
            "AttnResMerge requires {} on device, got {}",
            name,
            tensor->storage_type());
        TT_FATAL(tensor->device() == partial.device(), "AttnResMerge requires {} on the same device as partial", name);
        TT_FATAL(tensor->layout() == Layout::TILE, "AttnResMerge requires TILE layout for {}", name);
        TT_FATAL(!tensor->memory_config().is_sharded(), "AttnResMerge supports interleaved operands only, {}", name);
        TT_FATAL(
            tensor->padded_shape().rank() == 4,
            "AttnResMerge requires rank-4 operands, {} has rank {}",
            name,
            tensor->padded_shape().rank());
    }
    TT_FATAL(!args.output_mem_config.is_sharded(), "AttnResMerge supports an interleaved output only");

    const auto& partial_shape = partial.padded_shape();
    const auto& prefix_sum_shape = prefix_sum.padded_shape();
    // The live stream is one plane by construction, and only the partial batches:
    // there is a partial per read site but a single stream behind all of them.
    TT_FATAL(
        prefix_sum_shape[0] == 1 && prefix_sum_shape[1] == partial_shape[1] &&
            prefix_sum_shape[2] == partial_shape[2] && prefix_sum_shape[3] == partial_shape[3],
        "AttnResMerge requires an unbatched prefix_sum matching partial's plane, got {} against {}",
        prefix_sum_shape,
        partial_shape);

    // The reader derives a row's scalar tile from the output tile index assuming
    // the candidate dim is 1, so a multi-candidate caller would silently read the
    // wrong row. Dim 0 is the read-site axis and is selected, not matched.
    TT_FATAL(partial_shape[1] == 1, "AttnResMerge requires a candidate dim of 1, got {}", partial_shape[1]);
    TT_FATAL(
        partial_shape[0] == 1 || args.site < partial_shape[0],
        "AttnResMerge site {} is past partial's dim 0 of {}",
        args.site,
        partial_shape[0]);

    for (const auto& [tensor, name] : scalars) {
        TT_FATAL(
            tensor->logical_shape()[-1] == 1,
            "AttnResMerge {} must carry one scalar per row, i.e. a logical last dim of 1, got {}",
            name,
            tensor->logical_shape()[-1]);
        for (int i = 1; i < 3; ++i) {
            TT_FATAL(
                tensor->padded_shape()[i] == partial_shape[i],
                "AttnResMerge {} dim {} is {} but partial's is {}; the candidate and row dims must match",
                name,
                i,
                tensor->padded_shape()[i],
                partial_shape[i]);
        }
        // dim 0 is the read-site axis, so it is selected rather than matched — and
        // a scalar that carries a single plane is shared by every site, which is
        // what lets a per-site live_scores sit alongside a batched shift and mass.
        TT_FATAL(
            tensor->padded_shape()[0] == 1 || args.site < tensor->padded_shape()[0],
            "AttnResMerge site {} is past {}'s dim 0 of {}",
            args.site,
            name,
            tensor->padded_shape()[0]);
    }

    TT_FATAL(
        partial_shape[-1] % TILE_WIDTH == 0 && partial_shape[-2] % TILE_HEIGHT == 0,
        "AttnResMerge requires tile-aligned inner dims, got {} x {}",
        partial_shape[-2],
        partial_shape[-1]);
}

tt::tt_metal::TensorSpec AttnResMergeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // From the logical shape, not the padded one, so that tile padding stays
    // labelled as padding rather than becoming readable data.
    //
    // Dim 0 is the partial's read-site axis and `site` picks one of them, so the
    // output is a single plane however many the caller handed over.
    auto output_shape = tensor_args.partial.logical_shape();
    output_shape[0] = 1;
    return tt::tt_metal::TensorSpec(
        output_shape,
        operations::TensorLayout(
            tensor_args.partial.dtype(), operations::PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor AttnResMergeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.partial.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor attn_res_merge(
    const Tensor& partial,
    const Tensor& prefix_sum,
    const Tensor& shift,
    const Tensor& mass,
    const Tensor& live_scores,
    uint32_t site,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResMergeDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .site = site, .output_mem_config = output_mem_config, .compute_kernel_config = compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{
        .partial = partial, .prefix_sum = prefix_sum, .shift = shift, .mass = mass, .live_scores = live_scores};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
