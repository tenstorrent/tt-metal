// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_rope_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

void NlpCreateQkvHeadsRopeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& qkv = tensor_args.qkv;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;

    auto check = [](const Tensor& t, const char* name, bool allow_sharded = false) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(t.buffer() != nullptr, "{} must be allocated", name);
        TT_FATAL(t.layout() == Layout::TILE, "{} must be tilized", name);
        // qkv may be width-sharded (e.g. straight off matmul_decode, no interleaved scatter): the
        // reader's TensorAccessor resolves each logical tile-page to the right core regardless of
        // layout, so a sharded qkv is read transparently. cos/sin stay interleaved.
        const auto ml = t.memory_config().memory_layout();
        TT_FATAL(
            ml == TensorMemoryLayout::INTERLEAVED || (allow_sharded && ml == TensorMemoryLayout::WIDTH_SHARDED),
            "{} must be INTERLEAVED{}",
            name,
            allow_sharded ? " or WIDTH_SHARDED" : "");
    };
    check(qkv, "qkv", /*allow_sharded=*/true);
    check(cos, "cos");
    check(sin, "sin");

    uint32_t hd = args.head_dim;
    TT_FATAL(hd % (TILE_WIDTH * 2) == 0, "head_dim ({}) must be divisible by {}", hd, TILE_WIDTH * 2);
    TT_FATAL(
        qkv.padded_shape()[-2] == TILE_HEIGHT,
        "this op requires Ht == 1 (seq one tile row); got seq {}",
        qkv.padded_shape()[-2]);
    uint32_t expected_w = (args.num_q_heads + 2 * args.num_kv_heads) * hd;
    TT_FATAL(
        qkv.padded_shape()[-1] == expected_w,
        "qkv width ({}) must equal (num_q_heads + 2*num_kv_heads)*head_dim ({})",
        qkv.padded_shape()[-1],
        expected_w);

    TT_FATAL(cos.dtype() == sin.dtype(), "cos and sin dtypes must match");
    TT_FATAL(cos.padded_shape() == sin.padded_shape(), "cos and sin dims must match");
    TT_FATAL(
        cos.padded_shape()[0] == 1 && cos.padded_shape()[1] == 1 && cos.padded_shape()[-1] == hd,
        "cos/sin must be (1, 1, seq, head_dim={})",
        hd);
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "output memory config must be INTERLEAVED");
}

NlpCreateQkvHeadsRopeDeviceOperation::spec_return_value_t NlpCreateQkvHeadsRopeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& qkv = tensor_args.qkv;
    uint32_t seq = tt::round_up(args.seq_len, TILE_HEIGHT);
    auto layout =
        tt::tt_metal::TensorLayout(qkv.dtype(), tt::tt_metal::PageConfig(qkv.layout()), args.output_mem_config);
    auto make = [&](uint32_t heads) { return TensorSpec(ttnn::Shape({1, heads, seq, args.head_dim}), layout); };
    return {make(args.num_q_heads), make(args.num_kv_heads), make(args.num_kv_heads)};
}

NlpCreateQkvHeadsRopeDeviceOperation::tensor_return_value_t NlpCreateQkvHeadsRopeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.qkv.device();
    return {
        create_device_tensor(std::get<0>(specs), device),
        create_device_tensor(std::get<1>(specs), device),
        create_device_tensor(std::get<2>(specs), device)};
}

ttsl::hash::hash_t NlpCreateQkvHeadsRopeDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<NlpCreateQkvHeadsRopeDeviceOperation>(
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        args.seq_len,
        args.output_mem_config,
        tensor_args.qkv,
        tensor_args.cos,
        tensor_args.sin);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_rope(
    const Tensor& qkv,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t seq_len,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::NlpCreateQkvHeadsRopeDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .seq_len = seq_len,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{.qkv = qkv, .cos = cos, .sin = sin};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
