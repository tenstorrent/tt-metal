// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_spec_tloop_proto_device_operation.hpp"

#include <array>
#include <cmath>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr std::string_view kOp = "gdn_spec_tloop_proto";

void check_tiled(const Tensor& t, std::string_view name, std::initializer_list<DataType> dtypes) {
    using namespace kda_factory_detail;
    check_allocated_device_tensor(t, kOp, name);
    check_layout(t, Layout::TILE, kOp, name);
    check_interleaved(t, kOp, name);
    bool ok = false;
    for (auto d : dtypes) {
        ok = ok || t.dtype() == d;
    }
    TT_FATAL(ok, "{}: {} has unsupported dtype {}", kOp, name, t.dtype());
}
}  // namespace

GdnSpecTloopProtoOperation::program_factory_t GdnSpecTloopProtoOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GdnSpecTloopProtoProgramFactory{};
}

void GdnSpecTloopProtoOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    check_tiled(in.qkv, "qkv", {DataType::BFLOAT16});
    check_tiled(in.dt_bias, "dt_bias", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled(in.neg_exp_A, "neg_exp_A", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled(in.ring, "ring", {DataType::FLOAT32});
    check_tiled(in.weight, "weight", {DataType::BFLOAT16});
    for (const auto& [t, name] : std::array{
             std::pair{&in.dt_bias, "dt_bias"},
             std::pair{&in.neg_exp_A, "neg_exp_A"},
             std::pair{&in.ring, "ring"},
             std::pair{&in.weight, "weight"}}) {
        check_same_device(in.qkv, *t, kOp, name);
    }
    check_output_interleaved(a.output_mem_config, kOp);
    check_compute_config(a.compute_kernel_config, kOp);
    TT_FATAL(a.compute_kernel_config.fp32_dest_acc_en, "{}: fp32_dest_acc_en must be enabled", kOp);
    TT_FATAL(!a.compute_kernel_config.packer_l1_acc, "{}: packer_l1_acc is unsupported", kOp);
    TT_FATAL(
        a.output_dtype == DataType::FLOAT32 || a.output_dtype == DataType::BFLOAT16,
        "{}: output_dtype must be FLOAT32 or BFLOAT16",
        kOp);
    const uint32_t Nv = a.num_value_heads, Nk = a.num_key_heads, Dk = a.key_dim, Dv = a.value_dim;
    TT_FATAL(Nv > 0 && Nk > 0 && Nv % Nk == 0, "{}: num_value_heads must be a multiple of num_key_heads", kOp);
    TT_FATAL(2 * Nv <= tt::constants::TILE_WIDTH, "{}: a|b must fit one tile row (2*Nv <= 32)", kOp);
    TT_FATAL(
        Dk > 0 && Dv > 0 && Dk % tt::constants::TILE_WIDTH == 0 && Dv % tt::constants::TILE_WIDTH == 0,
        "{}: key_dim and value_dim must be tile aligned",
        kOp);
    TT_FATAL(std::isfinite(a.scale) && a.l2_epsilon > 0.0f && a.norm_epsilon > 0.0f, "{}: bad scale/epsilon", kOp);
    TT_FATAL(a.T >= 1 && a.B >= 1, "{}: need T >= 1, B >= 1", kOp);
    TT_FATAL(
        a.B * a.T <= tt::constants::TILE_HEIGHT || (tt::constants::TILE_HEIGHT % a.T == 0),
        "{}: a user's T rows must lie in one tile row (B*T <= 32 or 32 %% T == 0; T = {}, B = {})",
        kOp,
        a.T,
        a.B);
    TT_FATAL(
        (a.T % 2 == 0) || a.B == 1 || (a.opt_flags & 8u),
        "{}: T must be even unless B == 1 or opt_flags bit3 (single-row writes); T = {}, B = {}",
        kOp,
        a.T,
        a.B);
    TT_FATAL(a.s0_slot < a.T, "{}: s0_slot must be < T", kOp);
    const auto& qs = in.qkv.logical_shape();
    const uint32_t C = 2 * Nk * Dk + Nv * Dv;
    TT_FATAL(
        qs.rank() == 3 && qs[0] == 1 && qs[1] >= a.B * a.T && qs[2] >= a.qkvz_dim + 2 * Nv,
        "{}: qkv must be [1, R >= B*T, W >= qkvz_dim + 2*Nv] (got {})",
        kOp,
        qs);
    TT_FATAL(
        a.qkvz_dim == C + Nv * Dv && a.qkvz_dim % tt::constants::TILE_WIDTH == 0,
        "{}: qkvz_dim must equal 2*Nk*Dk + 2*Nv*Dv and be tile aligned",
        kOp);
    TT_FATAL(
        in.dt_bias.logical_volume() == Nv && in.neg_exp_A.logical_volume() == Nv,
        "{}: dt_bias / neg_exp_A volume must be Nv",
        kOp);
    const auto& rs = in.ring.logical_shape();
    TT_FATAL(
        rs.rank() == 3 && rs[0] == a.T * a.B * Nv && rs[1] == Dk && rs[2] == Dv,
        "{}: ring must be [T*B*Nv, Dk, Dv] (got {})",
        kOp,
        rs);
    TT_FATAL(in.weight.logical_volume() == Dv, "{}: weight volume must equal value_dim", kOp);
}

GdnSpecTloopProtoOperation::spec_return_value_t GdnSpecTloopProtoOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return TensorSpec(
        Shape({1, in.qkv.logical_shape()[-2], a.num_value_heads * a.value_dim}),
        TensorLayout(a.output_dtype, PageConfig(Layout::TILE), a.output_mem_config));
}

GdnSpecTloopProtoOperation::tensor_return_value_t GdnSpecTloopProtoOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(a, in), in.qkv.device());
}

Tensor gdn_spec_tloop_proto(
    const Tensor& qkv,
    const Tensor& dt_bias,
    const Tensor& neg_exp_A,
    const Tensor& ring,
    const Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t T,
    uint32_t B,
    uint32_t qkvz_dim,
    uint32_t s0_slot,
    float scale,
    float l2_epsilon,
    float norm_epsilon,
    bool row_batched,
    bool write_ring,
    uint32_t opt_flags,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype) {
    return ttnn::device_operation::launch<GdnSpecTloopProtoOperation>(
        GdnSpecTloopProtoParams{
            .num_value_heads = num_value_heads,
            .num_key_heads = num_key_heads,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .T = T,
            .B = B,
            .qkvz_dim = qkvz_dim,
            .s0_slot = s0_slot,
            .scale = scale,
            .l2_epsilon = l2_epsilon,
            .norm_epsilon = norm_epsilon,
            .row_batched = row_batched,
            .write_ring = write_ring,
            .opt_flags = opt_flags,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .compute_kernel_config = compute_kernel_config,
        },
        GdnSpecTloopProtoInputs{
            .qkv = qkv, .dt_bias = dt_bias, .neg_exp_A = neg_exp_A, .ring = ring, .weight = weight});
}

}  // namespace ttnn::experimental::prim
