// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_spec_step_device_operation.hpp"

#include <array>
#include <cmath>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr std::string_view kOp = "gdn_spec_step";

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

GdnSpecStepOperation::program_factory_t GdnSpecStepOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GdnSpecStepProgramFactory{};
}

void GdnSpecStepOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    constexpr uint32_t TW = tt::constants::TILE_WIDTH;
    check_tiled(in.qkvzab, "qkvzab", {DataType::BFLOAT16});
    check_tiled(in.win_a, "win_a", {DataType::BFLOAT16});
    check_tiled(in.win_b, "win_b", {DataType::BFLOAT16});
    check_tiled(in.ring, "ring", {DataType::FLOAT32});
    check_tiled(in.taps, "taps", {DataType::BFLOAT16});
    check_tiled(in.dt_bias, "dt_bias", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled(in.neg_exp_A, "neg_exp_A", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled(in.weight, "weight", {DataType::BFLOAT16});
    check_allocated_device_tensor(in.ctrl, kOp, "ctrl");
    check_layout(in.ctrl, Layout::ROW_MAJOR, kOp, "ctrl");
    check_interleaved(in.ctrl, kOp, "ctrl");
    TT_FATAL(
        in.ctrl.dtype() == DataType::UINT32 || in.ctrl.dtype() == DataType::INT32,
        "{}: ctrl must be UINT32/INT32 (got {})",
        kOp,
        in.ctrl.dtype());
    for (const auto& [t, name] : std::array{
             std::pair{&in.win_a, "win_a"},
             std::pair{&in.win_b, "win_b"},
             std::pair{&in.ring, "ring"},
             std::pair{&in.ctrl, "ctrl"},
             std::pair{&in.taps, "taps"},
             std::pair{&in.dt_bias, "dt_bias"},
             std::pair{&in.neg_exp_A, "neg_exp_A"},
             std::pair{&in.weight, "weight"}}) {
        check_same_device(in.qkvzab, *t, kOp, name);
    }
    TT_FATAL(in.win_a.buffer() != in.win_b.buffer(), "{}: win_a and win_b must be distinct buffers", kOp);
    check_output_interleaved(a.output_mem_config, kOp);
    check_compute_config(a.compute_kernel_config, kOp);
    TT_FATAL(a.compute_kernel_config.fp32_dest_acc_en, "{}: fp32_dest_acc_en must be enabled", kOp);
    TT_FATAL(!a.compute_kernel_config.packer_l1_acc, "{}: packer_l1_acc is unsupported", kOp);
    TT_FATAL(
        a.output_dtype == DataType::FLOAT32 || a.output_dtype == DataType::BFLOAT16,
        "{}: output_dtype must be FLOAT32 or BFLOAT16",
        kOp);
    const uint32_t Nv = a.num_value_heads, Nk = a.num_key_heads, Dk = a.key_dim, Dv = a.value_dim;
    const uint32_t T = a.T, B = a.B, K = a.conv_kernel;
    TT_FATAL(Nv > 0 && Nk > 0 && Nv % Nk == 0, "{}: num_value_heads must be a multiple of num_key_heads", kOp);
    // one core per (user, value head); the a|b gate pair may span two tiles (Nv = 24 at TP = 2), see the factory
    const auto grid = in.qkvzab.device()->compute_with_storage_grid_size();
    TT_FATAL(
        B * Nv <= static_cast<uint32_t>(grid.x * grid.y),
        "{}: B * num_value_heads = {} (user, head) items exceed the {}-core compute grid",
        kOp,
        B * Nv,
        grid.x * grid.y);
    TT_FATAL(Dk > 0 && Dv > 0 && Dk % TW == 0 && Dv % TW == 0, "{}: key_dim and value_dim must be tile aligned", kOp);
    TT_FATAL(std::isfinite(a.scale) && a.l2_epsilon > 0.0f && a.norm_epsilon > 0.0f, "{}: bad scale/epsilon", kOp);
    TT_FATAL(T >= 1 && B >= 1, "{}: need T >= 1, B >= 1", kOp);
    TT_FATAL(K >= 2 && K <= 4, "{}: conv_kernel must be in [2, 4] (got {})", kOp, K);
    const uint32_t Lw = K - 1 + T;
    TT_FATAL(Lw <= tt::constants::TILE_HEIGHT, "{}: window K-1+T = {} must fit one tile (<= 32)", kOp, Lw);
    TT_FATAL(
        B * T <= tt::constants::TILE_HEIGHT || (tt::constants::TILE_HEIGHT % T == 0),
        "{}: a user's T rows must lie in one tile row (B*T <= 32 or 32 %% T == 0; T = {}, B = {})",
        kOp,
        T,
        B);
    TT_FATAL(a.hnew_depth == 2 || a.hnew_depth == 4, "{}: hnew_depth must be 2 or 4", kOp);
    const uint32_t C = 2 * Nk * Dk + Nv * Dv;
    TT_FATAL(
        a.qkvz_dim == C + Nv * Dv && a.qkvz_dim % TW == 0,
        "{}: qkvz_dim must equal 2*Nk*Dk + 2*Nv*Dv and be tile aligned",
        kOp);
    const auto& qs = in.qkvzab.logical_shape();
    // R may exceed B*T only within the users' last tile row: the op zero-fills rows [B*T, round_up(B*T, 32)) of the
    // output, so 'rows outside [u*T, u*T + T) are exactly 0' holds for every R it accepts
    const uint32_t R_max =
        ((B * T + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    TT_FATAL(
        qs.rank() == 3 && qs[0] == 1 && qs[1] >= B * T && qs[1] <= R_max && qs[2] >= a.qkvz_dim + 2 * Nv,
        "{}: qkvzab must be [1, B*T <= R <= round_up(B*T, 32) = {}, W >= qkvz_dim + 2*Nv] (got {})",
        kOp,
        R_max,
        qs);
    for (const auto& [t, name] : std::array{std::pair{&in.win_a, "win_a"}, std::pair{&in.win_b, "win_b"}}) {
        const auto& ws = t->logical_shape();
        TT_FATAL(
            ws.rank() == 3 && ws[0] >= B && ws[1] >= Lw && ws[1] <= tt::constants::TILE_HEIGHT && ws[2] == C,
            "{}: {} must be [B' >= B, Lw <= L <= 32, C = {}] (got {})",
            kOp,
            name,
            C,
            ws);
    }
    const auto& ts = in.taps.logical_shape();
    TT_FATAL(
        ts.rank() == 3 && ts[0] == 1 && ts[1] >= K && ts[1] <= tt::constants::TILE_HEIGHT && ts[2] == C,
        "{}: taps must be [1, K <= rows <= 32, C = {}] with tap j in row j (got {})",
        kOp,
        C,
        ts);
    TT_FATAL(
        in.ring.logical_volume() >= static_cast<uint64_t>(T) * B * Nv * Dk * Dv && in.ring.logical_shape()[-1] == Dv &&
            in.ring.logical_shape()[-2] == Dk,
        "{}: ring must hold >= T*B*Nv blocks of [Dk, Dv] (got {})",
        kOp,
        in.ring.logical_shape());
    const auto& cs = in.ctrl.logical_shape();
    const uint32_t words = static_cast<uint32_t>(cs[-1]);
    TT_FATAL(
        static_cast<uint64_t>(words) == in.ctrl.logical_volume() && words >= 1 + B + B * Nv && (words * 4) % 64 == 0,
        "{}: ctrl must be one row-major page [.., N] with N >= 1 + B + B*Nv and N*4 %% 64 == 0 (got {})",
        kOp,
        cs);
    TT_FATAL(
        in.dt_bias.logical_volume() == Nv && in.neg_exp_A.logical_volume() == Nv,
        "{}: dt_bias / neg_exp_A volume must be Nv",
        kOp);
    TT_FATAL(in.weight.logical_volume() == Dv, "{}: weight volume must equal value_dim", kOp);
}

GdnSpecStepOperation::spec_return_value_t GdnSpecStepOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return TensorSpec(
        Shape({1, in.qkvzab.logical_shape()[-2], a.num_value_heads * a.value_dim}),
        TensorLayout(a.output_dtype, PageConfig(Layout::TILE), a.output_mem_config));
}

GdnSpecStepOperation::tensor_return_value_t GdnSpecStepOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(a, in), in.qkvzab.device());
}

Tensor gdn_spec_step(
    const Tensor& qkvzab,
    const Tensor& win_a,
    const Tensor& win_b,
    const Tensor& ring,
    const Tensor& ctrl,
    const Tensor& taps,
    const Tensor& dt_bias,
    const Tensor& neg_exp_A,
    const Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t T,
    uint32_t B,
    uint32_t conv_kernel,
    uint32_t qkvz_dim,
    float scale,
    float l2_epsilon,
    float norm_epsilon,
    uint32_t hnew_depth,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype) {
    return ttnn::device_operation::launch<GdnSpecStepOperation>(
        GdnSpecStepParams{
            .num_value_heads = num_value_heads,
            .num_key_heads = num_key_heads,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .T = T,
            .B = B,
            .conv_kernel = conv_kernel,
            .qkvz_dim = qkvz_dim,
            .scale = scale,
            .l2_epsilon = l2_epsilon,
            .norm_epsilon = norm_epsilon,
            .hnew_depth = hnew_depth,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .compute_kernel_config = compute_kernel_config,
        },
        GdnSpecStepInputs{
            .qkvzab = qkvzab,
            .win_a = win_a,
            .win_b = win_b,
            .ring = ring,
            .ctrl = ctrl,
            .taps = taps,
            .dt_bias = dt_bias,
            .neg_exp_A = neg_exp_A,
            .weight = weight});
}

}  // namespace ttnn::experimental::prim
