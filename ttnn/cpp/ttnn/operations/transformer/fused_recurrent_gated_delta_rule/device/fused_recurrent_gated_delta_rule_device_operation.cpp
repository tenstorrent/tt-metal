// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_gated_delta_rule_device_operation.hpp"

#include <initializer_list>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

FusedRecurrentGatedDeltaRuleDeviceOperation::program_factory_t
FusedRecurrentGatedDeltaRuleDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return FusedRecurrentGatedDeltaRuleProgramFactory{};
}

void FusedRecurrentGatedDeltaRuleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    // Attribute contract (the kernel derives every tile count from these).
    TT_FATAL(attrs.T >= 1, "fused_recurrent_gated_delta_rule: T must be >= 1, got {}", attrs.T);
    TT_FATAL(attrs.BH >= 1, "fused_recurrent_gated_delta_rule: B*HV must be >= 1, got {}", attrs.BH);
    TT_FATAL(
        attrs.key_dim >= TILE_WIDTH && attrs.key_dim % TILE_WIDTH == 0,
        "fused_recurrent_gated_delta_rule: key_dim must be a non-zero multiple of {}, got {}",
        TILE_WIDTH,
        attrs.key_dim);
    TT_FATAL(
        attrs.val_dim >= TILE_WIDTH && attrs.val_dim % TILE_WIDTH == 0,
        "fused_recurrent_gated_delta_rule: val_dim must be a non-zero multiple of {}, got {}",
        TILE_WIDTH,
        attrs.val_dim);
    const uint32_t BHT = attrs.BH * attrs.T;
    const uint32_t K = attrs.key_dim;
    const uint32_t V = attrs.val_dim;

    // Tensor contract: fp32 TILE interleaved (DRAM or L1), all on q's device, exact logical shapes. The
    // reader/writer index pages as [BH*T,1,K] / [BH*T,1,V] / [BH*T,1,1] / [BH,K,V]; a mismatch
    // would read past the buffer instead of failing, so every dim is checked here.
    auto* device = in.q.device();

    // The recurrent state is carried across tokens in DST (see rank1_update in the compute
    // kernel), which requires fp32 DST registers.
    const bool fp32_dest_acc_en =
        std::get<2>(get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config));
    TT_FATAL(
        fp32_dest_acc_en,
        "fused_recurrent_gated_delta_rule: requires fp32_dest_acc_en (the recurrent state is carried in fp32 DST)");

    auto check = [&](const Tensor& t, const char* name, std::initializer_list<uint32_t> expected) {
        TT_FATAL(
            t.storage_type() == StorageType::DEVICE && t.is_allocated(),
            "fused_recurrent_gated_delta_rule: {} must be allocated on device",
            name);
        TT_FATAL(t.device() == device, "fused_recurrent_gated_delta_rule: {} must be on the same device as q", name);
        // DRAM or L1 is fine (the kernels address pages through TensorAccessor); sharding is not.
        TT_FATAL(!t.is_sharded(), "fused_recurrent_gated_delta_rule: {} must be interleaved, not sharded", name);
        TT_FATAL(t.layout() == Layout::TILE, "fused_recurrent_gated_delta_rule: {} must be TILE layout", name);
        TT_FATAL(
            t.dtype() == DataType::FLOAT32,
            "fused_recurrent_gated_delta_rule: {} must be fp32, got {}",
            name,
            t.dtype());
        const auto& shape = t.logical_shape();
        TT_FATAL(
            shape.rank() == expected.size(),
            "fused_recurrent_gated_delta_rule: {} must be rank {}, got rank {}",
            name,
            expected.size(),
            shape.rank());
        size_t i = 0;
        for (uint32_t e : expected) {
            TT_FATAL(
                static_cast<uint32_t>(shape[i]) == e,
                "fused_recurrent_gated_delta_rule: {} dim[{}] must be {}, got {}",
                name,
                i,
                e,
                shape[i]);
            ++i;
        }
    };
    // The layer tensors: TILE, 32x32 tiles, interleaved, bf16 or fp32 (q, k, v of one dtype), on q's device. The
    // reader addresses their pages from these dims, so every dim is checked.
    auto check_input = [&](const Tensor& t, const char* name, std::initializer_list<uint32_t> expected) {
        TT_FATAL(
            t.storage_type() == StorageType::DEVICE && t.is_allocated(),
            "fused_recurrent_gated_delta_rule: {} must be allocated on device",
            name);
        TT_FATAL(t.device() == device, "fused_recurrent_gated_delta_rule: {} must be on the same device as q", name);
        TT_FATAL(!t.is_sharded(), "fused_recurrent_gated_delta_rule: {} must be interleaved, not sharded", name);
        TT_FATAL(t.layout() == Layout::TILE, "fused_recurrent_gated_delta_rule: {} must be TILE layout", name);
        TT_FATAL(
            t.dtype() == DataType::FLOAT32 || t.dtype() == DataType::BFLOAT16,
            "fused_recurrent_gated_delta_rule: {} must be bf16 or fp32, got {}",
            name,
            t.dtype());
        const auto tile_shape = t.tensor_spec().tile().get_tile_shape();
        TT_FATAL(
            tile_shape[0] == TILE_HEIGHT && tile_shape[1] == TILE_WIDTH,
            "fused_recurrent_gated_delta_rule: {} must use 32x32 tiles, got {}x{}",
            name,
            tile_shape[0],
            tile_shape[1]);
        const auto& shape = t.logical_shape();
        TT_FATAL(
            shape.rank() == expected.size(),
            "fused_recurrent_gated_delta_rule: {} must be rank {}, got rank {}",
            name,
            expected.size(),
            shape.rank());
        size_t i = 0;
        for (uint32_t e : expected) {
            TT_FATAL(
                static_cast<uint32_t>(shape[i]) == e,
                "fused_recurrent_gated_delta_rule: {} dim[{}] must be {}, got {}",
                name,
                i,
                e,
                shape[i]);
            ++i;
        }
    };
    TT_FATAL(
        attrs.H >= 1 && attrs.HV % attrs.H == 0 && attrs.B * attrs.HV == attrs.BH,
        "fused_recurrent_gated_delta_rule: HV ({}) must be a multiple of H ({}) and B*HV ({}*{}) must equal BH ({})",
        attrs.HV,
        attrs.H,
        attrs.B,
        attrs.HV,
        attrs.BH);
    check_input(in.q, "q", {attrs.B, attrs.T, attrs.H, K});
    check_input(in.k, "k", {attrs.B, attrs.T, attrs.H, K});
    check_input(in.v, "v", {attrs.B, attrs.T, attrs.HV, V});
    check_input(in.g, "g", {attrs.B, attrs.T, attrs.HV});  // the log decay; the kernel takes exp(g)
    check_input(in.beta, "beta", {attrs.B, attrs.T, attrs.HV});
    TT_FATAL(
        in.k.dtype() == in.q.dtype() && in.v.dtype() == in.q.dtype(),
        "fused_recurrent_gated_delta_rule: q, k and v must share one dtype, got {} {} {}",
        in.q.dtype(),
        in.k.dtype(),
        in.v.dtype());
    // Ring mode: initial_state is the [BH*T,K,V] ring (also the in-place state output); otherwise [BH,K,V].
    const bool ring = in.initial_state_block_idx.has_value();
    TT_FATAL(
        !attrs.inplace_final_state || (in.initial_state.has_value() && !ring && !attrs.output_per_token_state),
        "fused_recurrent_gated_delta_rule: inplace_final_state requires initial_state and excludes ring mode and the "
        "per-token state");
    if (in.output_tensor.has_value()) {
        const Tensor& o = *in.output_tensor;
        const auto& os = o.logical_shape();
        TT_FATAL(
            o.storage_type() == StorageType::DEVICE && o.is_allocated() && o.device() == device && !o.is_sharded() &&
                o.layout() == Layout::TILE && o.dtype() == in.q.dtype() && os.rank() == 4 &&
                static_cast<uint32_t>(os[0]) == attrs.B && static_cast<uint32_t>(os[1]) == attrs.T &&
                static_cast<uint32_t>(os[2]) == attrs.HV && static_cast<uint32_t>(os[3]) == V,
            "fused_recurrent_gated_delta_rule: output_tensor must be [B, T, HV, V] in the input dtype, TILE, "
            "interleaved, on q's device, got {} {}",
            o.dtype(),
            os);
    }
    if (in.initial_state.has_value() && !ring) {
        // The caller's state: [Bs, HV, K, V] with Bs*HV >= BH; head h's tiles are its pages h*kv .. h*kv + kv - 1, so
        // the reader and the writer address it as [BH, K, V]. With inplace_final_state it is also the state output
        // (the reader reads head h's pages before the writer's only write to them, after the last token).
        const Tensor& s = *in.initial_state;
        const auto& ss = s.logical_shape();
        TT_FATAL(
            s.storage_type() == StorageType::DEVICE && s.is_allocated() && s.device() == device && !s.is_sharded() &&
                s.layout() == Layout::TILE && s.dtype() == DataType::FLOAT32 && ss.rank() == 4 &&
                static_cast<uint32_t>(ss[0]) * static_cast<uint32_t>(ss[1]) >= attrs.BH &&
                static_cast<uint32_t>(ss[2]) == K && static_cast<uint32_t>(ss[3]) == V,
            "fused_recurrent_gated_delta_rule: initial_state must be fp32 TILE interleaved [Bs, HV, K, V] with "
            "Bs*HV >= {} on q's device, got {} {}",
            attrs.BH,
            s.dtype(),
            ss);
    } else if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state (the ring)", {BHT, K, V});
    }
    if (ring) {
        TT_FATAL(
            attrs.output_per_token_state,
            "fused_recurrent_gated_delta_rule: initial_state_block_idx requires output_per_token_state");
        const Tensor& idx = *in.initial_state_block_idx;
        TT_FATAL(
            idx.storage_type() == StorageType::DEVICE && idx.is_allocated(),
            "fused_recurrent_gated_delta_rule: initial_state_block_idx must be allocated on device");
        TT_FATAL(
            idx.device() == device,
            "fused_recurrent_gated_delta_rule: initial_state_block_idx must be on the same device as q");
        TT_FATAL(!idx.is_sharded(), "fused_recurrent_gated_delta_rule: initial_state_block_idx must be interleaved");
        TT_FATAL(
            idx.layout() == Layout::ROW_MAJOR,
            "fused_recurrent_gated_delta_rule: initial_state_block_idx must be ROW_MAJOR");
        TT_FATAL(
            idx.dtype() == DataType::UINT32 || idx.dtype() == DataType::INT32,
            "fused_recurrent_gated_delta_rule: initial_state_block_idx must be uint32 or int32, got {}",
            idx.dtype());
        const auto& ishape = idx.logical_shape();
        const bool shape_ok = (ishape.rank() == 1 && static_cast<uint32_t>(ishape[0]) == attrs.BH) ||
                              (ishape.rank() == 2 && ishape[0] == 1 && static_cast<uint32_t>(ishape[1]) == attrs.BH);
        TT_FATAL(
            shape_ok,
            "fused_recurrent_gated_delta_rule: initial_state_block_idx must be [{}] or [1,{}], got {}",
            attrs.BH,
            attrs.BH,
            ishape);
    }
}

FusedRecurrentGatedDeltaRuleDeviceOperation::spec_return_value_t
FusedRecurrentGatedDeltaRuleDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    // o: [B, T, HV, V] in the input dtype (the caller's output_tensor's spec when given).
    const auto o_spec = in.output_tensor.has_value()
                            ? in.output_tensor->tensor_spec()
                            : tt::tt_metal::TensorSpec(
                                  ttnn::Shape({attrs.B, attrs.T, attrs.HV, attrs.val_dim}),
                                  TensorLayout(in.q.dtype(), PageConfig(Layout::TILE), attrs.output_mem_config));
    if (attrs.inplace_final_state) {
        TT_FATAL(
            in.initial_state.has_value(),
            "fused_recurrent_gated_delta_rule: inplace_final_state requires initial_state");
        return {o_spec, in.initial_state->tensor_spec()};
    }
    if (in.initial_state_block_idx.has_value()) {
        // create_output_tensors runs before validate_on_program_cache_miss, so check before dereferencing.
        TT_FATAL(
            in.initial_state.has_value(),
            "fused_recurrent_gated_delta_rule: initial_state_block_idx requires initial_state (the ring)");
        return {o_spec, in.initial_state->tensor_spec()};
    }
    // state: per token [B, T, HV, K, V] (verify slots), else final [B, HV, K, V].
    ttnn::Shape s_shape = attrs.output_per_token_state
                              ? ttnn::Shape({attrs.B, attrs.T, attrs.HV, attrs.key_dim, attrs.val_dim})
                              : ttnn::Shape({attrs.B, attrs.HV, attrs.key_dim, attrs.val_dim});
    return {o_spec, tt::tt_metal::TensorSpec(s_shape, layout)};
}

FusedRecurrentGatedDeltaRuleDeviceOperation::tensor_return_value_t
FusedRecurrentGatedDeltaRuleDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    // o: the caller's tensor when given (the writer writes into it), else a new one.
    if (in.output_tensor.has_value()) {
        outs.push_back(*in.output_tensor);
    } else {
        outs.push_back(create_device_tensor(specs[0], device));
    }
    // Ring mode and the in-place final state: the state output aliases the caller's tensor (the adapter allows an
    // output buffer to alias an input).
    if (in.initial_state_block_idx.has_value() || attrs.inplace_final_state) {
        outs.push_back(*in.initial_state);
    } else {
        outs.push_back(create_device_tensor(specs[1], device));
    }
    return outs;
}

std::vector<Tensor> fused_recurrent_gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state,
    const std::optional<Tensor>& initial_state_block_idx,
    const std::optional<Tensor>& output_tensor,
    uint32_t B,
    uint32_t H,
    uint32_t HV,
    uint32_t T,
    bool output_final_state,
    bool output_per_token_state,
    bool inplace_final_state,
    bool use_qk_l2norm_in_kernel,
    float scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = FusedRecurrentGatedDeltaRuleDeviceOperation;

    const auto& q_shape = q.logical_shape();  // [B, T, H, K]
    const auto& v_shape = v.logical_shape();  // [B, T, HV, V]
    TT_FATAL(T >= 1, "fused_recurrent_gated_delta_rule: T must be >= 1, got {}", T);
    TT_FATAL(
        q_shape.rank() == 4 && v_shape.rank() == 4,
        "fused_recurrent_gated_delta_rule: q and v must be rank 4, got ranks {} and {}",
        q_shape.rank(),
        v_shape.rank());

    auto attrs = OperationType::operation_attributes_t{
        .B = B,
        .H = H,
        .HV = HV,
        .BH = B * HV,
        .T = T,
        .key_dim = q_shape[3],
        .val_dim = v_shape[3],
        .output_final_state = output_final_state,
        .output_per_token_state = output_per_token_state,
        .inplace_final_state = inplace_final_state,
        .use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel,
        .scale = scale,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .initial_state = initial_state,
        .initial_state_block_idx = initial_state_block_idx,
        .output_tensor = output_tensor,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
