// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_decode_step_device_operation.hpp"

#include <array>
#include <cmath>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr std::string_view kOp = "gdn_decode_step";

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

GdnDecodeStepOperation::program_factory_t GdnDecodeStepOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GdnDecodeStepProgramFactory{};
}

void GdnDecodeStepOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    check_tiled(in.qkv, "qkv", {DataType::BFLOAT16});
    check_tiled(in.beta, a.fuse_conv ? "dt_bias" : "beta", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled(in.g, a.fuse_conv ? "neg_exp_A" : "g", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled(in.state, "state", {DataType::FLOAT32});
    check_tiled(in.weight, "weight", {DataType::BFLOAT16});
    for (const auto& [t, name] : std::array{
             std::pair{&in.beta, "beta"},
             std::pair{&in.g, "g"},
             std::pair{&in.state, "state"},
             std::pair{&in.weight, "weight"}}) {
        check_same_device(in.qkv, *t, kOp, name);
    }
    check_output_interleaved(a.output_mem_config, kOp);
    check_compute_config(a.compute_kernel_config, kOp);
    TT_FATAL(
        a.compute_kernel_config.fp32_dest_acc_en,
        "{}: fp32_dest_acc_en must be enabled (32-bit state accumulation and in-DST transpose)",
        kOp);
    TT_FATAL(!a.compute_kernel_config.packer_l1_acc, "{}: packer_l1_acc is unsupported", kOp);
    TT_FATAL(
        a.output_dtype == DataType::FLOAT32 || a.output_dtype == DataType::BFLOAT16,
        "{}: output_dtype must be FLOAT32 or BFLOAT16",
        kOp);
    const uint32_t Nv = a.num_value_heads, Nk = a.num_key_heads, Dk = a.key_dim, Dv = a.value_dim;
    TT_FATAL(Nv > 0 && Nk > 0 && Nv % Nk == 0, "{}: num_value_heads must be a multiple of num_key_heads", kOp);
    TT_FATAL(Nv <= tt::constants::TILE_WIDTH, "{}: num_value_heads must fit one tile row (<= 32)", kOp);
    TT_FATAL(
        Dk > 0 && Dv > 0 && Dk % tt::constants::TILE_WIDTH == 0 && Dv % tt::constants::TILE_WIDTH == 0,
        "{}: key_dim and value_dim must be tile aligned",
        kOp);
    TT_FATAL(std::isfinite(a.scale) && a.l2_epsilon > 0.0f && a.norm_epsilon > 0.0f, "{}: bad scale/epsilon", kOp);
    const auto& qs = in.qkv.logical_shape();
    const uint32_t C = 2 * Nk * Dk + Nv * Dv;
    if (a.fuse_conv) {
        TT_FATAL(
            qs.rank() == 3 && qs[0] == 1 && qs[1] == 1 && qs[2] >= a.qkvz_dim + 2 * Nv,
            "{}: fused-conv qkv (projection row) must be [1, 1, W >= qkvz_dim + 2*Nv] (got {})",
            kOp,
            qs);
        TT_FATAL(
            a.qkvz_dim == C + Nv * Dv && a.qkvz_dim % tt::constants::TILE_WIDTH == 0 &&
                2 * Nv <= tt::constants::TILE_WIDTH,
            "{}: fused-conv needs qkvz_dim == 2*Nk*Dk + 2*Nv*Dv (tile aligned) and 2*Nv <= 32 (a|b in one tile)",
            kOp);
        TT_FATAL(
            in.conv_states.size() == 4 && in.conv_taps.size() == 4,
            "{}: fused-conv needs 4 conv states and 4 taps",
            kOp);
        for (uint32_t j = 0; j < 4; ++j) {
            check_tiled(in.conv_states[j], "conv_state", {DataType::BFLOAT16});
            check_tiled(in.conv_taps[j], "conv_tap", {DataType::BFLOAT16});
            check_same_device(in.qkv, in.conv_states[j], kOp, "conv_state");
            check_same_device(in.qkv, in.conv_taps[j], kOp, "conv_tap");
            const auto& cs = in.conv_states[j].logical_shape();
            TT_FATAL(
                cs.rank() == 3 && cs[0] == 1 && cs[1] == 1 && cs[2] == C,
                "{}: conv_state must be [1, 1, C] (got {})",
                kOp,
                cs);
            TT_FATAL(in.conv_taps[j].logical_volume() == C, "{}: conv_tap volume must equal C", kOp);
        }
        TT_FATAL(
            in.beta.logical_volume() == Nv && in.g.logical_volume() == Nv,
            "{}: dt_bias / neg_exp_A volume must be Nv",
            kOp);
    } else {
        TT_FATAL(
            qs.rank() == 3 && qs[0] == 1 && qs[1] == 1 && qs[2] == C,
            "{}: qkv must be [1, 1, 2*Nk*Dk + Nv*Dv] (got {})",
            kOp,
            qs);
        for (const auto& [t, name] : std::array{std::pair{&in.beta, "beta"}, std::pair{&in.g, "g"}}) {
            const auto& s = t->logical_shape();
            TT_FATAL(
                s.rank() == 3 && s[0] == 1 && s[1] == 1 && s[2] == Nv,
                "{}: {} must be [1, 1, Nv] (got {})",
                kOp,
                name,
                s);
        }
    }
    const auto& ss = in.state.logical_shape();
    TT_FATAL(
        ss.rank() == 4 && ss[0] == 1 && ss[1] == Nv && ss[2] == Dk && ss[3] == Dv,
        "{}: state must be [1, Nv, Dk, Dv] (got {})",
        kOp,
        ss);
    TT_FATAL(in.weight.logical_volume() == Dv, "{}: weight volume must equal value_dim", kOp);
}

GdnDecodeStepOperation::spec_return_value_t GdnDecodeStepOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return TensorSpec(
        Shape({1, 1, a.num_value_heads * a.value_dim}),
        TensorLayout(a.output_dtype, PageConfig(Layout::TILE), a.output_mem_config));
}

GdnDecodeStepOperation::tensor_return_value_t GdnDecodeStepOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(a, in), in.qkv.device());
}

Tensor gdn_decode_step(
    const Tensor& qkv,
    const Tensor& beta,
    const Tensor& g,
    const Tensor& state,
    const Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    float scale,
    float l2_epsilon,
    float norm_epsilon,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype,
    const std::vector<Tensor>& conv_states,
    const std::vector<Tensor>& conv_taps,
    uint32_t qkvz_dim) {
    return ttnn::device_operation::launch<GdnDecodeStepOperation>(
        GdnDecodeStepParams{
            .num_value_heads = num_value_heads,
            .num_key_heads = num_key_heads,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .scale = scale,
            .l2_epsilon = l2_epsilon,
            .norm_epsilon = norm_epsilon,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .compute_kernel_config = compute_kernel_config,
            .fuse_conv = !conv_states.empty(),
            .qkvz_dim = qkvz_dim,
        },
        GdnDecodeStepInputs{
            .qkv = qkv,
            .beta = beta,
            .g = g,
            .state = state,
            .weight = weight,
            .conv_states = conv_states,
            .conv_taps = conv_taps});
}

}  // namespace ttnn::experimental::prim
