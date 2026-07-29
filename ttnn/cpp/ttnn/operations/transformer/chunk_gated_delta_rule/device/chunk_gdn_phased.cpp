// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gdn_phased.hpp"

#include <cstdlib>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
void check(const Tensor& t, const char* name, DataType dt) {
    TT_FATAL(t.layout() == Layout::TILE, "chunk_gdn: {} must be TILE layout", name);
    TT_FATAL(t.dtype() == dt, "chunk_gdn: {} has wrong dtype", name);
    TT_FATAL(t.buffer() != nullptr, "chunk_gdn: {} must be on device", name);
}

void check_intermediate(const Tensor& t, const char* name, bool allow_bf16) {
    TT_FATAL(t.layout() == Layout::TILE, "chunk_gdn: {} must be TILE layout", name);
    TT_FATAL(
        t.dtype() == DataType::FLOAT32 || (allow_bf16 && t.dtype() == DataType::BFLOAT16),
        "chunk_gdn: {} must be FLOAT32{}",
        name,
        allow_bf16 ? " or BFLOAT16" : "");
    TT_FATAL(t.buffer() != nullptr, "chunk_gdn: {} must be on device", name);
}
}  // namespace

// ---------------------------------------------------------------------------
// PREP
// ---------------------------------------------------------------------------
ChunkGdnPrepOperation::program_factory_t ChunkGdnPrepOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChunkGdnPrepProgramFactory{};
}

void ChunkGdnPrepOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    check(in.q, "q", DataType::BFLOAT16);
    check(in.k, "k", DataType::BFLOAT16);
    check(in.v, "v", DataType::BFLOAT16);  // flat [B,T,HV*V] when attrs.v_flat; else [BH,NC,C,V]
    if (attrs.v_flat) {
        TT_FATAL(attrs.HV > 0, "v_flat requires HV > 0");
        const auto& vs = in.v.logical_shape();
        TT_FATAL(vs.rank() == 3, "v_flat expects a flat [B,T,HV*V] v (got rank {})", vs.rank());
        TT_FATAL(vs[2] == attrs.HV * attrs.val_dim, "v_flat width {} != HV*V ({}*{})", vs[2], attrs.HV, attrs.val_dim);
    }
    if (attrs.qk_flat) {
        TT_FATAL(attrs.Hk > 0, "qk_flat requires Hk > 0");
        const auto& qsf = in.q.logical_shape();
        TT_FATAL(qsf.rank() == 3, "qk_flat expects a flat [B,T,Hk*K] q (got rank {})", qsf.rank());
        TT_FATAL(
            qsf[2] == attrs.Hk * attrs.key_dim, "qk_flat width {} != Hk*K ({}*{})", qsf[2], attrs.Hk, attrs.key_dim);
        TT_FATAL(attrs.qk_norm, "qk_flat requires qk_norm (flat q/k are unnormalized; norm is in-kernel)");
    }
    if (attrs.vector_gate) {
        check_intermediate(in.g, "g", true);
    } else {
        check(in.g, "g", DataType::FLOAT32);
    }
    if (attrs.g_flat) {
        TT_FATAL(attrs.vector_gate && attrs.HV > 0, "g_flat requires vector_gate and HV > 0");
        const auto& gs = in.g.logical_shape();
        TT_FATAL(gs.rank() == 3, "g_flat expects a flat [B,T,HV*K] g (got rank {})", gs.rank());
        TT_FATAL(gs[2] == attrs.HV * attrs.key_dim, "g_flat width {} != HV*K ({}*{})", gs[2], attrs.HV, attrs.key_dim);
    }
    check(in.beta, "beta", DataType::FLOAT32);
    check(in.eye_c, "eye_c", DataType::FLOAT32);
    check(in.tril_c, "tril_c", DataType::FLOAT32);
    check(in.ones_c, "ones_c", DataType::FLOAT32);
    check(in.masks_c, "masks_c", DataType::FLOAT32);
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
}

ChunkGdnPrepOperation::spec_return_value_t ChunkGdnPrepOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    constexpr uint32_t allowed_bf16_mask = 0x37;  // v_beta, kd, q_decay, k_dec_t, dl
    TT_FATAL(
        (attrs.output_bf16_mask & ~allowed_bf16_mask) == 0,
        "unsupported KDA prep BF16 mask 0x{:x}",
        attrs.output_bf16_mask);
    const auto spec = [&](const ttnn::Shape& s, uint32_t output_index) {
        const auto dtype = (attrs.output_bf16_mask & (1u << output_index)) ? DataType::BFLOAT16 : DataType::FLOAT32;
        return tt::tt_metal::TensorSpec(s, TensorLayout(dtype, PageConfig(Layout::TILE), attrs.output_mem_config));
    };
    const uint32_t BH = attrs.BH, NC = attrs.num_chunks, C = attrs.chunk_size, K = attrs.key_dim, V = attrs.val_dim;
    return {
        spec(ttnn::Shape({BH, NC, C, V}), 0),                          // v_beta
        spec(ttnn::Shape({BH, NC, C, K}), 1),                          // kd
        spec(ttnn::Shape({BH, NC, C, K}), 2),                          // q_decay
        spec(ttnn::Shape({BH, NC, C, C}), 3),                          // intra
        spec(ttnn::Shape({BH, NC, K, C}), 4),                          // k_dec_t
        spec(ttnn::Shape({BH, NC, attrs.vector_gate ? K : 1, 1}), 5),  // dl
        spec(ttnn::Shape({BH, NC, C, C}), 6),                          // t_inv
    };
}

ChunkGdnPrepOperation::tensor_return_value_t ChunkGdnPrepOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::vector<Tensor> chunk_gdn_prep(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const Tensor& masks_c,
    uint32_t chunk_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool v_flat,
    uint32_t HV,
    bool qk_norm,
    float scale,
    bool qk_flat,
    uint32_t Hk,
    bool g_flat,
    bool vector_gate,
    uint32_t output_bf16_mask) {
    const auto& q_shape = q.logical_shape();  // [BH,NC,C,K] head-major, or flat [B,T,Hk*K] when qk_flat
    const auto& v_shape = v.logical_shape();  // [BH,NC,C,V] head-major, or flat [B,T,HV*V] when v_flat
    // Derive dims. Head-major q gives BH/NC/K directly; flat q [B,T,Hk*K] gives B/T, so BH=B*HV,
    // NC=T/chunk (pad==0 required), K=flat_width/Hk. val_dim = v_shape[3] or v_flat width / HV.
    const uint32_t BH = qk_flat ? (q_shape[0] * HV) : q_shape[0];
    const uint32_t num_chunks = qk_flat ? (q_shape[1] / chunk_size) : q_shape[1];
    const uint32_t key_dim = qk_flat ? (q_shape[2] / Hk) : q_shape[3];
    const uint32_t val_dim = v_flat ? (v_shape[2] / HV) : v_shape[3];
    auto attrs = ChunkGdnPrepOperation::operation_attributes_t{
        .BH = BH,
        .num_chunks = num_chunks,
        .chunk_size = chunk_size,
        .key_dim = key_dim,
        .val_dim = val_dim,
        .v_flat = v_flat,
        .HV = HV,
        .qk_flat = qk_flat,
        .Hk = Hk,
        .g_flat = g_flat,
        .qk_norm = qk_norm,
        .scale = scale,
        .vector_gate = vector_gate,
        .output_bf16_mask = output_bf16_mask,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = ChunkGdnPrepOperation::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .eye_c = eye_c,
        .tril_c = tril_c,
        .ones_c = ones_c,
        .masks_c = masks_c};
    return ttnn::device_operation::launch<ChunkGdnPrepOperation>(attrs, tensor_args);
}

// ---------------------------------------------------------------------------
// SCAN
// ---------------------------------------------------------------------------
ChunkGdnScanOperation::program_factory_t ChunkGdnScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChunkGdnScanProgramFactory{};
}

void ChunkGdnScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    check_intermediate(in.v_beta, "v_beta", attrs.vector_gate);
    check_intermediate(in.kd, "kd", attrs.vector_gate);
    check_intermediate(in.q_decay, "q_decay", attrs.vector_gate);
    check(in.intra, "intra", DataType::FLOAT32);
    check_intermediate(in.k_dec_t, "k_dec_t", attrs.vector_gate);
    check_intermediate(in.dl, "dl", attrs.vector_gate);
    check(in.t_inv, "t_inv", DataType::FLOAT32);
    if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state", DataType::FLOAT32);
    }
    if (in.identity_tile.has_value()) {
        check(*in.identity_tile, "identity_tile", DataType::FLOAT32);
        TT_FATAL(attrs.key_dim == attrs.val_dim, "identity initial state requires K == V");
    }
    TT_FATAL(
        !attrs.summary_pair || (attrs.state_only && in.identity_tile.has_value()),
        "summary_pair requires state_only and an identity tile");
    TT_FATAL(
        !(in.initial_state.has_value() && in.identity_tile.has_value()),
        "initial_state and identity_tile are mutually exclusive");
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
    TT_FATAL(in.rms_gate.has_value() == in.rms_weight.has_value(), "fused RMS requires both gate and weight");
    if (in.rms_gate.has_value()) {
        TT_FATAL(attrs.chunk_size == TILE_HEIGHT, "fused RMS requires 32-token chunks");
        check(*in.rms_gate, "rms_gate", DataType::BFLOAT16);
        check(*in.rms_weight, "rms_weight", DataType::BFLOAT16);
        TT_FATAL(
            attrs.vector_gate && attrs.num_heads > 0 && attrs.BH % attrs.num_heads == 0,
            "fused RMS requires KDA heads");
        const auto& gs = in.rms_gate->logical_shape();
        TT_FATAL(
            gs.rank() == 3 && gs[0] == attrs.BH / attrs.num_heads && gs[1] == attrs.num_chunks * attrs.chunk_size &&
                gs[2] == attrs.num_heads * attrs.val_dim,
            "fused RMS gate shape mismatch");
        TT_FATAL(in.rms_weight->logical_volume() == attrs.val_dim, "fused RMS weight width mismatch");
    }
}

ChunkGdnScanOperation::spec_return_value_t ChunkGdnScanOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    // o is fp32; the recurrent final state is fp32 too. (A bf16 o output — feeding a bf16 attention
    // result into every GDN layer — measurably degraded full-model quality, so it was removed; the
    // seq path also keeps o fp32.)
    const auto o_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto s_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    ttnn::Shape o_shape =
        attrs.summary_pair
            ? ttnn::Shape({attrs.BH, attrs.key_dim, attrs.val_dim})
            : (attrs.state_only ? ttnn::Shape({1, 1, 32, 32})
                                : ttnn::Shape({attrs.BH, attrs.num_chunks, attrs.chunk_size, attrs.val_dim}));
    ttnn::Shape s_shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    std::vector<tt::tt_metal::TensorSpec> specs{
        tt::tt_metal::TensorSpec(o_shape, o_layout), tt::tt_metal::TensorSpec(s_shape, s_layout)};
    if (attrs.fused_rms) {
        specs.emplace_back(
            ttnn::Shape(
                {attrs.BH / attrs.num_heads, attrs.num_chunks * attrs.chunk_size, attrs.num_heads * attrs.val_dim}),
            o_layout);
    }
    return specs;
}

ChunkGdnScanOperation::tensor_return_value_t ChunkGdnScanOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.v_beta.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::vector<Tensor> chunk_gdn_scan(
    const Tensor& v_beta,
    const Tensor& kd,
    const Tensor& q_decay,
    const Tensor& intra,
    const Tensor& k_dec_t,
    const Tensor& dl,
    const Tensor& t_inv,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool vector_gate,
    bool state_only,
    const std::optional<Tensor>& identity_tile,
    const std::optional<Tensor>& rms_gate,
    const std::optional<Tensor>& rms_weight,
    uint32_t num_heads,
    float rms_epsilon,
    bool summary_pair) {
    const auto& vb_shape = v_beta.logical_shape();  // [BH, NC, C, V]
    const auto& kd_shape = kd.logical_shape();      // [BH, NC, C, K]
    auto attrs = ChunkGdnScanOperation::operation_attributes_t{
        .BH = vb_shape[0],
        .num_chunks = vb_shape[1],
        .chunk_size = chunk_size,
        .key_dim = kd_shape[3],
        .val_dim = vb_shape[3],
        .has_initial_state = initial_state.has_value(),
        .identity_initial_state = identity_tile.has_value(),
        .output_final_state = output_final_state,
        .state_only = state_only,
        .summary_pair = summary_pair,
        .vector_gate = vector_gate,
        .fused_rms = rms_gate.has_value(),
        .num_heads = num_heads,
        .rms_epsilon = rms_epsilon,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = ChunkGdnScanOperation::tensor_args_t{
        .v_beta = v_beta,
        .kd = kd,
        .q_decay = q_decay,
        .intra = intra,
        .k_dec_t = k_dec_t,
        .dl = dl,
        .t_inv = t_inv,
        .initial_state = initial_state,
        .identity_tile = identity_tile,
        .rms_gate = rms_gate,
        .rms_weight = rms_weight};
    return ttnn::device_operation::launch<ChunkGdnScanOperation>(attrs, tensor_args);
}

// ---------------------------------------------------------------------------
// KDA GROUPED AFFINE PREFIX
// ---------------------------------------------------------------------------
KdaAffinePrefixOperation::program_factory_t KdaAffinePrefixOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaAffinePrefixProgramFactory{};
}

void KdaAffinePrefixOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check(in.transform_a, "transform_a", DataType::FLOAT32);
    check(in.transform_b, "transform_b", DataType::FLOAT32);
    const auto& as = in.transform_a.logical_shape();
    const auto& bs = in.transform_b.logical_shape();
    TT_FATAL(as.rank() == 3 && bs.rank() == 3, "KDA affine prefix expects rank-3 transforms");
    TT_FATAL(attrs.groups_per_head > 0, "groups_per_head must be positive");
    TT_FATAL(as[0] == attrs.BH * attrs.groups_per_head, "transform_a leading dimension mismatch");
    TT_FATAL(bs[0] == as[0], "transform_a/transform_b leading dimensions must match");
    TT_FATAL(as[1] == attrs.key_dim && as[2] == attrs.key_dim, "transform_a must be [BH*G,K,K]");
    TT_FATAL(bs[1] == attrs.key_dim && bs[2] == attrs.val_dim, "transform_b must be [BH*G,K,V]");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be tile aligned");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be tile aligned");
    if (attrs.compose_only) {
        TT_FATAL(!in.initial_state.has_value(), "compose-only affine prefix does not take an initial state");
    } else {
        TT_FATAL(in.initial_state.has_value(), "affine prefix requires an initial state");
        check(*in.initial_state, "initial_state", DataType::FLOAT32);
        const auto& ss = in.initial_state->logical_shape();
        TT_FATAL(ss.rank() == 3, "KDA affine prefix expects a rank-3 initial state");
        TT_FATAL(ss[0] == attrs.BH && ss[1] == attrs.key_dim && ss[2] == attrs.val_dim, "initial_state shape mismatch");
    }
}

KdaAffinePrefixOperation::spec_return_value_t KdaAffinePrefixOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = [&](const Shape& shape) {
        return TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config));
    };
    if (attrs.compose_only) {
        return {
            layout(Shape({attrs.BH, attrs.key_dim, attrs.key_dim})),
            layout(Shape({attrs.BH, attrs.key_dim, attrs.val_dim}))};
    }
    return {layout(Shape({attrs.BH * attrs.groups_per_head, attrs.key_dim, attrs.val_dim}))};
}

KdaAffinePrefixOperation::tensor_return_value_t KdaAffinePrefixOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.transform_a.device()));
    }
    return outputs;
}

namespace {
KdaAffinePrefixParams affine_prefix_params(
    const Tensor& transform_a,
    const Tensor& transform_b,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool compose_only) {
    const auto& as = transform_a.logical_shape();
    const auto& bs = transform_b.logical_shape();
    TT_FATAL(groups_per_head > 0 && as[0] % groups_per_head == 0, "invalid affine group count");
    return {
        .BH = static_cast<uint32_t>(as[0]) / groups_per_head,
        .groups_per_head = groups_per_head,
        .key_dim = static_cast<uint32_t>(as[1]),
        .val_dim = static_cast<uint32_t>(bs[2]),
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .compose_only = compose_only};
}
}  // namespace

Tensor kda_affine_prefix(
    const Tensor& transform_a,
    const Tensor& transform_b,
    const Tensor& initial_state,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto results = ttnn::device_operation::launch<KdaAffinePrefixOperation>(
        affine_prefix_params(
            transform_a,
            transform_b,
            groups_per_head,
            output_mem_config,
            compute_kernel_config,
            /*compose_only=*/false),
        KdaAffinePrefixInputs{.transform_a = transform_a, .transform_b = transform_b, .initial_state = initial_state});
    return results[0];
}

std::pair<Tensor, Tensor> kda_affine_compose(
    const Tensor& transform_a,
    const Tensor& transform_b,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto results = ttnn::device_operation::launch<KdaAffinePrefixOperation>(
        affine_prefix_params(
            transform_a,
            transform_b,
            groups_per_head,
            output_mem_config,
            compute_kernel_config,
            /*compose_only=*/true),
        KdaAffinePrefixInputs{.transform_a = transform_a, .transform_b = transform_b, .initial_state = std::nullopt});
    return {results[0], results[1]};
}

KdaGatedRmsOperation::program_factory_t KdaGatedRmsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaGatedRmsProgramFactory{};
}

void KdaGatedRmsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check(in.input, "input", DataType::FLOAT32);
    check(in.gate, "gate", DataType::BFLOAT16);
    check(in.weight, "weight", DataType::BFLOAT16);
    const auto& xs = in.input.logical_shape();
    const auto& gs = in.gate.logical_shape();
    TT_FATAL(xs.rank() == 3, "KDA gated RMS input must be [B*H,T,V]");
    TT_FATAL(gs.rank() == 3, "KDA gated RMS gate must be [B,T,H*V]");
    TT_FATAL(
        xs[0] == attrs.batch * attrs.num_heads && xs[1] == attrs.sequence && xs[2] == attrs.value_dim,
        "KDA gated RMS input shape does not match attributes");
    TT_FATAL(
        gs[0] == attrs.batch && gs[1] == attrs.sequence && gs[2] == attrs.num_heads * attrs.value_dim,
        "KDA gated RMS gate shape does not match attributes");
    TT_FATAL(in.weight.logical_volume() == attrs.value_dim, "KDA gated RMS weight volume must equal V");
    TT_FATAL(attrs.sequence % TILE_HEIGHT == 0, "KDA gated RMS sequence must be tile aligned");
    TT_FATAL(attrs.value_dim % TILE_WIDTH == 0, "KDA gated RMS value_dim must be tile aligned");
}

KdaGatedRmsOperation::spec_return_value_t KdaGatedRmsOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    return {TensorSpec(
        Shape({attrs.batch, attrs.sequence, attrs.num_heads * attrs.value_dim}),
        TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config))};
}

KdaGatedRmsOperation::tensor_return_value_t KdaGatedRmsOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {create_device_tensor(specs[0], in.input.device())};
}

Tensor kda_gated_rms_norm(
    const Tensor& input,
    const Tensor& gate,
    const Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& xs = input.logical_shape();
    TT_FATAL(xs.rank() == 3, "KDA gated RMS input must be [B*H,T,V]");
    TT_FATAL(num_heads > 0, "KDA gated RMS num_heads must be positive");
    TT_FATAL(xs[0] % num_heads == 0, "KDA gated RMS leading dimension must be divisible by num_heads");
    const uint32_t batch = xs[0] / num_heads;
    auto results = ttnn::device_operation::launch<KdaGatedRmsOperation>(
        KdaGatedRmsParams{
            .batch = batch,
            .num_heads = num_heads,
            .sequence = static_cast<uint32_t>(xs[1]),
            .value_dim = static_cast<uint32_t>(xs[2]),
            .epsilon = epsilon,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        KdaGatedRmsInputs{.input = input, .gate = gate, .weight = weight});
    return results[0];
}

KdaCausalConvOperation::program_factory_t KdaCausalConvOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaCausalConvProgramFactory{};
}

void KdaCausalConvOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    TT_FATAL(in.input.layout() == Layout::ROW_MAJOR, "kda_causal_conv1d_split: input must be ROW_MAJOR");
    TT_FATAL(in.input.dtype() == DataType::BFLOAT16, "kda_causal_conv1d_split: input must be BFLOAT16");
    TT_FATAL(in.input.buffer() != nullptr, "kda_causal_conv1d_split: input must be on device");
    TT_FATAL(in.state.layout() == Layout::ROW_MAJOR, "kda_causal_conv1d_split: state must be ROW_MAJOR");
    TT_FATAL(in.state.dtype() == DataType::BFLOAT16, "kda_causal_conv1d_split: state must be BFLOAT16");
    TT_FATAL(in.state.buffer() != nullptr, "kda_causal_conv1d_split: state must be on device");
    check(in.tap0, "tap0", DataType::BFLOAT16);
    check(in.tap1, "tap1", DataType::BFLOAT16);
    check(in.tap2, "tap2", DataType::BFLOAT16);
    check(in.tap3, "tap3", DataType::BFLOAT16);
    const uint32_t channels = attrs.q_width + attrs.k_width + attrs.v_width;
    const auto& xs = in.input.logical_shape();
    const auto& ss = in.state.logical_shape();
    TT_FATAL(xs.rank() == 3 && xs[0] == 1 && xs[1] == attrs.sequence && xs[2] == channels, "input must be [1,T,Q+K+V]");
    TT_FATAL(ss.rank() == 3 && ss[0] == 1 && ss[1] == 3 && ss[2] == channels, "state must be [1,3,Q+K+V]");
    TT_FATAL(attrs.sequence % 32 == 0, "sequence must be tile aligned");
    TT_FATAL(
        attrs.q_width % 32 == 0 && attrs.k_width % 32 == 0 && attrs.v_width % 32 == 0,
        "Q/K/V widths must be tile aligned");
    TT_FATAL(in.tap0.logical_volume() == channels, "tap0 width mismatch");
    TT_FATAL(in.tap1.logical_volume() == channels, "tap1 width mismatch");
    TT_FATAL(in.tap2.logical_volume() == channels, "tap2 width mismatch");
    TT_FATAL(in.tap3.logical_volume() == channels, "tap3 width mismatch");
}

KdaCausalConvOperation::spec_return_value_t KdaCausalConvOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    return {
        TensorSpec(Shape({1, attrs.sequence, attrs.q_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.k_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.v_width}), layout)};
}

KdaCausalConvOperation::tensor_return_value_t KdaCausalConvOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.input.device()),
        create_device_tensor(specs[1], in.input.device()),
        create_device_tensor(specs[2], in.input.device())};
}

std::vector<Tensor> kda_causal_conv1d_split(
    const Tensor& input,
    const Tensor& state,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& shape = input.logical_shape();
    return ttnn::device_operation::launch<KdaCausalConvOperation>(
        KdaCausalConvParams{
            .sequence = static_cast<uint32_t>(shape[1]),
            .q_width = q_width,
            .k_width = k_width,
            .v_width = v_width,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        KdaCausalConvInputs{.input = input, .state = state, .tap0 = tap0, .tap1 = tap1, .tap2 = tap2, .tap3 = tap3});
}

}  // namespace ttnn::prim
