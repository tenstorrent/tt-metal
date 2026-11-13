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
    // the circular buffers and page reads are sized for 32x32 tiles
    const auto& tile = t.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "{}: {} must use 32x32 tiles (got {}x{})",
        kOp,
        name,
        tile.get_height(),
        tile.get_width());
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
    // the convolution history comes from the window pair or, with conv_hist, from packed history pages
    const bool hist_paged = in.conv_hist.has_value();
    TT_FATAL(
        hist_paged ? (!in.win_a.has_value() && !in.win_b.has_value()) : (in.win_a.has_value() && in.win_b.has_value()),
        "{}: pass either win_a and win_b (window pair) or conv_hist with ring_page_idx (history pages), not both "
        "(got win_a {}, win_b {}, conv_hist {})",
        kOp,
        in.win_a.has_value(),
        in.win_b.has_value(),
        hist_paged);
    if (!hist_paged) {
        check_tiled(*in.win_a, "win_a", {DataType::BFLOAT16});
        check_tiled(*in.win_b, "win_b", {DataType::BFLOAT16});
        check_same_device(in.qkvzab, *in.win_a, kOp, "win_a");
        check_same_device(in.qkvzab, *in.win_b, kOp, "win_b");
        TT_FATAL(in.win_a->buffer() != in.win_b->buffer(), "{}: win_a and win_b must be distinct buffers", kOp);
    }
    for (const auto& [t, name] : std::array{
             std::pair{&in.ring, "ring"},
             std::pair{&in.ctrl, "ctrl"},
             std::pair{&in.taps, "taps"},
             std::pair{&in.dt_bias, "dt_bias"},
             std::pair{&in.neg_exp_A, "neg_exp_A"},
             std::pair{&in.weight, "weight"}}) {
        check_same_device(in.qkvzab, *t, kOp, name);
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
    const uint32_t T = a.T, B = a.B, K = a.conv_kernel;
    TT_FATAL(Nv > 0 && Nk > 0 && Nv % Nk == 0, "{}: num_value_heads must be a multiple of num_key_heads", kOp);
    // one core per (user, value head)
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
    TT_FATAL(a.value_blocks <= 4, "{}: value_blocks must be in [0, 4] (got {})", kOp, a.value_blocks);
    TT_FATAL(
        (a.kernel_opts & ~static_cast<uint32_t>(kGdnSpecOptAll)) == 0,
        "{}: unsupported kernel_opts bits {:#x} (supported mask {:#x})",
        kOp,
        a.kernel_opts & ~static_cast<uint32_t>(kGdnSpecOptAll),
        static_cast<uint32_t>(kGdnSpecOptAll));
    TT_FATAL(
        a.kernel_opts == 0 || (a.fast_mode && a.fp32_state),
        "{}: kernel_opts need fast_mode and fp32_state (got {:#x})",
        kOp,
        a.kernel_opts);
    TT_FATAL(K >= 2 && K <= 4, "{}: conv_kernel must be in [2, 4] (got {})", kOp, K);
    const uint32_t Lw = K - 1 + T;
    TT_FATAL(Lw <= tt::constants::TILE_HEIGHT, "{}: window K-1+T = {} must fit one tile (<= 32)", kOp, Lw);
    TT_FATAL(
        B * T <= tt::constants::TILE_HEIGHT || (tt::constants::TILE_HEIGHT % T == 0),
        "{}: a user's T rows must lie in one tile row (B*T <= 32 or 32 %% T == 0; T = {}, B = {})",
        kOp,
        T,
        B);
    const uint32_t C = 2 * Nk * Dk + Nv * Dv;
    TT_FATAL(
        a.qkvz_dim == C + Nv * Dv && a.qkvz_dim % TW == 0 && 2 * Nv <= TW,
        "{}: qkvz_dim must equal 2*Nk*Dk + 2*Nv*Dv (tile aligned) and 2*Nv <= 32 (a|b in one tile)",
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
        if (!t->has_value()) {
            continue;
        }
        const auto& ws = (*t)->logical_shape();
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
    if (in.ring_page_idx.has_value()) {
        const Tensor& idx = *in.ring_page_idx;
        check_allocated_device_tensor(idx, kOp, "ring_page_idx");
        check_layout(idx, Layout::ROW_MAJOR, kOp, "ring_page_idx");
        check_interleaved(idx, kOp, "ring_page_idx");
        check_same_device(in.qkvzab, idx, kOp, "ring_page_idx");
        TT_FATAL(
            idx.dtype() == DataType::INT32 || idx.dtype() == DataType::UINT32,
            "{}: ring_page_idx must be INT32 or UINT32 (got {})",
            kOp,
            idx.dtype());
        const auto& is = idx.logical_shape();
        const uint32_t idx_words = (in.conv_hist.has_value() ? 2 : 1) * T * Nv;
        TT_FATAL(
            is.rank() == 2 && is[0] >= B && is[1] == idx_words,
            "{}: ring_page_idx must be [>= B, {}] (T*Nv, twice that with conv_hist; got {}, B = {}, T = {}, Nv = {})",
            kOp,
            idx_words,
            is,
            B,
            T,
            Nv);
        if (in.conv_hist.has_value()) {
            const Tensor& hist = *in.conv_hist;
            check_tiled(hist, "conv_hist", {DataType::BFLOAT16});
            check_same_device(in.qkvzab, hist, kOp, "conv_hist");
            TT_FATAL(K == 4, "{}: conv_hist holds gdn_decode_step's 4-slot history; conv_kernel must be 4", kOp);
            // a head's [q | k | v] chunks take tile rows 2c and 2c + 1 of one packed page
            TT_FATAL(
                2 * ((2 * Dk + Dv) / TW) <= tt::constants::TILE_HEIGHT,
                "{}: a head's [q | k | v] row ({} channels) does not fit one packed history tile",
                kOp,
                2 * Dk + Dv);
            // a held head reads its placeholder history from pages 0..3
            TT_FATAL(hist.buffer()->num_pages() >= 4, "{}: conv_hist must hold at least 4 pages", kOp);
        }
        if (in.ring_pool2.has_value()) {
            check_tiled(*in.ring_pool2, "ring_pool2", {DataType::FLOAT32});
            check_same_device(in.qkvzab, *in.ring_pool2, kOp, "ring_pool2");
            // the kernels address both pools with one accessor type and ring's page size (same_layout_accessor)
            TT_FATAL(
                in.ring_pool2->buffer()->buffer_type() == in.ring.buffer()->buffer_type(),
                "{}: ring_pool2 must be in the same memory (DRAM / L1) as ring",
                kOp);
            TT_FATAL(
                in.ring_pool2->buffer()->aligned_page_size() == in.ring.buffer()->aligned_page_size(),
                "{}: ring_pool2 must have the aligned page size of ring ({} B, got {} B)",
                kOp,
                in.ring.buffer()->aligned_page_size(),
                in.ring_pool2->buffer()->aligned_page_size());
            TT_FATAL(
                in.ring_pool2->buffer()->num_pages() >= static_cast<uint64_t>(Dk / TW) * (Dv / TW),
                "{}: ring_pool2 must hold at least Dk * Dv / 1024 tiles",
                kOp);
        }
        // a held head reads ring block u*Nv + h as a placeholder state, so blocks [0, B*Nv) must exist
        TT_FATAL(
            in.ring.buffer()->num_pages() >= static_cast<uint64_t>(B) * Nv * (Dk / TW) * (Dv / TW),
            "{}: ring pool must hold at least B * Nv * Dk * Dv / 1024 tiles (got {})",
            kOp,
            in.ring.buffer()->num_pages());
    } else {
        TT_FATAL(!in.ring_pool2.has_value(), "{}: ring_pool2 needs ring_page_idx", kOp);
        TT_FATAL(!in.conv_hist.has_value(), "{}: conv_hist needs ring_page_idx", kOp);
        TT_FATAL(
            in.ring.logical_volume() >= static_cast<uint64_t>(T) * B * Nv * Dk * Dv &&
                in.ring.logical_shape()[-1] == Dv && in.ring.logical_shape()[-2] == Dk,
            "{}: ring must hold >= T*B*Nv blocks of [Dk, Dv] (got {})",
            kOp,
            in.ring.logical_shape());
    }
    const auto& cs = in.ctrl.logical_shape();
    const uint32_t words = static_cast<uint32_t>(cs[-1]);
    // with conv_hist the ctrl page also carries the start history page of every (u, h)
    const uint32_t ctrl_words = 1 + B + (in.conv_hist.has_value() ? 2 : 1) * B * Nv;
    TT_FATAL(
        static_cast<uint64_t>(words) == in.ctrl.logical_volume() && words >= ctrl_words && (words * 4) % 64 == 0,
        "{}: ctrl must be one row-major page [.., N] with N >= {} and N*4 %% 64 == 0 (got {})",
        kOp,
        ctrl_words,
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
    const std::optional<Tensor>& win_a,
    const std::optional<Tensor>& win_b,
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
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype,
    bool fast_mode,
    bool fp32_state,
    uint32_t value_blocks,
    uint32_t kernel_opts,
    const std::optional<Tensor>& ring_page_idx,
    const std::optional<Tensor>& ring_pool2,
    const std::optional<Tensor>& conv_hist) {
    TT_FATAL(!fp32_state || fast_mode, "{}: fp32_state needs fast_mode", kOp);
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
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .compute_kernel_config = compute_kernel_config,
            .fast_mode = fast_mode,
            .fp32_state = fp32_state,
            .value_blocks = value_blocks,
            .kernel_opts = kernel_opts,
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
            .weight = weight,
            .ring_page_idx = ring_page_idx,
            .ring_pool2 = ring_pool2,
            .conv_hist = conv_hist});
}

}  // namespace ttnn::experimental::prim
