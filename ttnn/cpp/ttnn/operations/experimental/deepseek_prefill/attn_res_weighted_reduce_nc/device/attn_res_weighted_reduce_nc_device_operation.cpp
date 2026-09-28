// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc/device/attn_res_weighted_reduce_nc_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

void AttnResWeightedReduceNCDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;

    // The compute config defaults to HiFi4 with fp32 dest accumulation, which is only
    // correct on Blackhole; elsewhere the op compiles, runs, and returns silently wrong
    // values. Reject the device rather than let a caller reach that path.
    const tt::ARCH arch = tt::tt_metal::hal::get_arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "AttnResWeightedReduceNC is only supported on Blackhole, got {}", arch);

    // The input is BFLOAT16 only: the MAC path is `mul_tiles_bcast_cols` with
    // acc_to_dest, and that is the one numeric configuration gated against a
    // reference. Widening it here would ship an untested path.
    //
    // The weight also takes FLOAT32, because callers that compute it in fp32
    // shouldn't have to spend a typecast to hand it over — AttnRes runs its whole
    // score chain in fp32 for exactly the accuracy reason that would make
    // downcasting at this boundary wrong. The CBs are sized from each tensor's own
    // dtype, so a mixed pair needs nothing from the program factory.
    operations::check_tensor(input, "AttnResWeightedReduceNC", "input", {DataType::BFLOAT16});
    operations::check_tensor(weight, "AttnResWeightedReduceNC", "weight", {DataType::BFLOAT16, DataType::FLOAT32});

    TT_FATAL(input.device() == weight.device(), "AttnResWeightedReduceNC requires input and weight on the same device");
    TT_FATAL(
        !input.memory_config().is_sharded() && !weight.memory_config().is_sharded() &&
            !args.output_mem_config.is_sharded(),
        "AttnResWeightedReduceNC supports interleaved operands only");

    // dim is deliberately narrow. The reader and the compute kernel each derive
    // the token-tile row from the output tile index using a formula that assumes
    // the reduced axis is dim 1; dim 0 is a small change to both and is simply
    // not written or tested yet.
    TT_FATAL(args.dim == 1, "AttnResWeightedReduceNC supports dim == 1 only, got {}", args.dim);

    const auto& input_shape = input.padded_shape();
    const auto& weight_shape = weight.padded_shape();
    TT_FATAL(
        input_shape.rank() == 4 && weight_shape.rank() == 4,
        "AttnResWeightedReduceNC requires rank-4 operands, got input rank {} and weight rank {}",
        input_shape.rank(),
        weight_shape.rank());

    TT_FATAL(
        weight.logical_shape()[-1] == 1,
        "AttnResWeightedReduceNC weight must carry one scalar per row, i.e. a logical last dim of 1, got {}",
        weight.logical_shape()[-1]);
    // Logical, not just padded. The row dim is tile-padded, so two different token
    // counts in one tile bucket — 100 and 120, both padding to 128 — compare equal
    // padded. The kernel would then weight the shorter operand's padding into rows
    // the output still reports as logical, and return them as data.
    const auto& input_logical = input.logical_shape();
    const auto& weight_logical = weight.logical_shape();
    for (int i = 1; i < 3; ++i) {
        TT_FATAL(
            input_logical[i] == weight_logical[i] && input_shape[i] == weight_shape[i],
            "AttnResWeightedReduceNC weight dim {} is {} ({} padded) but input's is {} ({} padded); the candidate "
            "and row dims must match",
            i,
            weight_logical[i],
            weight_shape[i],
            input_logical[i],
            input_shape[i]);
    }
    // The weight's dim 0 is the output's batch, so the input carries none. The
    // kernel's whole structure rests on one input serving every weight set; an
    // input batch would need a separate read per set and buy nothing.
    TT_FATAL(
        input_shape[0] == 1,
        "AttnResWeightedReduceNC takes an unbatched input, got dim 0 of {}; the batch comes from the weight",
        input_shape[0]);
}

tt::tt_metal::TensorSpec AttnResWeightedReduceNCDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // keepdim=true, matching fast_reduce_nc: the reduced axis becomes 1 rather
    // than disappearing, so callers can chain without a reshape.
    //
    // From the logical shape, not the padded one. fast_reduce_nc builds this from
    // padded_shape, which hands a caller with 100 tokens an output claiming 128
    // logical rows — the 28 rows of tile padding become part of the tensor, and
    // whatever the input's padding held is now readable data. Taking the logical
    // shape leaves the padding labelled as padding; the kernels are unaffected,
    // since they work in tiles either way.
    auto output_shape = tensor_args.input.logical_shape();
    output_shape[args.dim] = 1;
    output_shape[0] = tensor_args.weight.logical_shape()[0];
    return tt::tt_metal::TensorSpec(
        output_shape,
        operations::TensorLayout(
            tensor_args.input.dtype(), operations::PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor AttnResWeightedReduceNCDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor attn_res_weighted_reduce_nc(
    const Tensor& input,
    const Tensor& weight,
    int32_t dim,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResWeightedReduceNCDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dim = dim, .output_mem_config = output_mem_config, .compute_kernel_config = compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{.input = input, .weight = weight};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
