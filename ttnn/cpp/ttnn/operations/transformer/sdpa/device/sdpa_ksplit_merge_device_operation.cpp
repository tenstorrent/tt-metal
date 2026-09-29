// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_ksplit_merge_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

void SDPAKSplitMergeDeviceOperation::validate_on_program_cache_miss(
    const SDPAKSplitMergeParams& args, const SDPAKSplitMergeInputs& tensor_args) {
    const auto& o = tensor_args.partial_output;
    const auto& st = tensor_args.partial_stats;
    const uint32_t S = args.k_split;
    TT_FATAL(
        S >= 1 && S <= SDPA_KSPLIT_MERGE_MAX_SPLIT,
        "sdpa_k_split_merge: k_split must be in [1, {}], got {}",
        SDPA_KSPLIT_MERGE_MAX_SPLIT,
        S);
    for (const auto* t : {&o, &st}) {
        TT_FATAL(t->storage_type() == StorageType::DEVICE, "sdpa_k_split_merge: inputs must be on device");
        TT_FATAL(t->buffer() != nullptr, "sdpa_k_split_merge: inputs must be allocated");
        TT_FATAL(t->layout() == Layout::TILE, "sdpa_k_split_merge: inputs must be tiled");
        TT_FATAL(t->dtype() == DataType::BFLOAT16, "sdpa_k_split_merge: inputs must be bfloat16");
        TT_FATAL(
            t->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "sdpa_k_split_merge: inputs must be interleaved");
        TT_FATAL(t->padded_shape().rank() == 4, "sdpa_k_split_merge: inputs must be rank 4");
    }
    const auto& os = o.padded_shape();
    const auto& ss = st.padded_shape();
    TT_FATAL(
        os[1] % S == 0, "sdpa_k_split_merge: partial_output dim 1 ({}) must be a multiple of k_split {}", os[1], S);
    TT_FATAL(
        ss[0] == os[0] && ss[1] == os[1],
        "sdpa_k_split_merge: partial_stats [B, k_split * NH] ({}, {}) must match partial_output ({}, {})",
        ss[0],
        ss[1],
        os[0],
        os[1]);
    TT_FATAL(
        ss[3] == tt::constants::TILE_WIDTH,
        "sdpa_k_split_merge: partial_stats must be one tile wide (the 32 per-column running-sum partials), got {}",
        ss[3]);
    TT_FATAL(
        ss[2] % (2 * tt::constants::TILE_HEIGHT) == 0 && ss[2] / 2 >= os[2],
        "sdpa_k_split_merge: partial_stats rows ({}) must be 2 x a tile multiple >= the output rows ({})",
        ss[2],
        os[2]);
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "sdpa_k_split_merge: output must be interleaved");
}

TensorSpec SDPAKSplitMergeDeviceOperation::compute_output_specs(
    const SDPAKSplitMergeParams& args, const SDPAKSplitMergeInputs& tensor_args) {
    auto shape = tensor_args.partial_output.logical_shape();
    shape[1] /= args.k_split;
    return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor SDPAKSplitMergeDeviceOperation::create_output_tensors(
    const SDPAKSplitMergeParams& args, const SDPAKSplitMergeInputs& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.partial_output.device());
}

Tensor sdpa_k_split_merge(
    const Tensor& partial_output,
    const Tensor& partial_stats,
    uint32_t k_split,
    float scale,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = SDPAKSplitMergeDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .k_split = k_split,
            .scale = scale,
            .output_mem_config = memory_config.value_or(partial_output.memory_config()),
        },
        OperationType::tensor_args_t{.partial_output = partial_output, .partial_stats = partial_stats});
}

}  // namespace ttnn::prim
