// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <utility>

#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa_numerics.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa_recipe.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"

namespace ttnn::transformer {

namespace {
// Empty joint (0 elements) == "no joint". Normalize to nullopt so self-attention callers passing
// zero-length dummy joints don't create duplicate input Buffer*s -> resolve_bindings() bails and the
// WorkloadDescriptor cache-hit path freezes stale addresses (#45452 / #45391). L=0 either way, so
// numerically identical.
std::optional<ttnn::Tensor> drop_if_empty(const std::optional<ttnn::Tensor>& t) {
    if (t.has_value() && t->logical_shape().volume() == 0) {
        return std::nullopt;
    }
    return t;
}
}  // namespace

ttnn::Tensor scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& attention_sink,
    const std::optional<ttnn::Tensor>& cu_window_seqlens,
    uint32_t windowed_q_token_offset,
    const std::optional<ttnn::Tensor>& windowed_q_token_offset_tensor,
    std::optional<SDPAPrecision> precision,
    bool inputs_prepared) {
    if (precision) {
        namespace numeric = operations::transformer::sdpa::detail;
        TT_FATAL(input_tensor_q.storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
        TT_FATAL(
            !is_causal && !attn_mask && !sliding_window_size && !attention_sink && !cu_window_seqlens &&
                windowed_q_token_offset == 0 && !windowed_q_token_offset_tensor,
            "Named SDPA recipes currently support dense noncausal, unmasked attention only");
        TT_FATAL(!memory_config || *memory_config == DRAM_MEMORY_CONFIG, "SDPA recipes require DRAM output");
        const auto policy = numeric::resolve_recipe_policy(
            input_tensor_q, input_tensor_k, *precision, inputs_prepared, scale, compute_kernel_config, program_config);
        return numeric::run_recipe(input_tensor_q, input_tensor_k, input_tensor_v, policy, program_config);
    }
    TT_FATAL(!inputs_prepared, "inputs_prepared is meaningful only with an explicit LOW_PRECISION recipe");
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    // Omitting precision preserves the existing numerical and dispatch contract.
    auto numerics = operations::transformer::sdpa::detail::resolve_numerics(
        input_tensor_q.device()->arch(),
        std::nullopt,
        compute_kernel_config,
        program_config ? program_config->exp_approx_mode : std::nullopt);
    auto kernel_config_val = numerics.compute;

    // PyTorch semantics: softmax(Q·Kᵀ * scale + mask) · V, where `scale` applies
    // to Q·Kᵀ only and the mask is added unscaled.
    //
    // The compute kernel folds `scale` into the softmax exponent as a
    // performance optimization:
    //     exp((QK + mask - row_max) * scale)
    //   = exp(QK*scale + mask*scale - row_max*scale)
    // which scales the mask along with QK, diverging from PyTorch semantics.
    //
    // Pre-multiply the mask by 1/scale so the kernel's subsequent *scale
    // restores the original mask magnitude inside softmax. QK remains scaled
    // exactly once.
    //
    // Windowed mode synthesizes a {0, -inf} block-diagonal mask on-device from cu_window_seqlens;
    // pre-scaling is unnecessary (0/-inf are scale-invariant), so attn_mask is left empty.
    std::optional<ttnn::Tensor> effective_mask = attn_mask;
    if (attn_mask.has_value()) {
        const float effective_scale =
            scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1])));
        if (effective_scale != 1.0f) {
            effective_mask = ttnn::multiply(attn_mask.value(), 1.0f / effective_scale);
        }
    }

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        effective_mask,
        std::nullopt,  // page_table
        attention_sink,
        is_causal,
        scale,
        sliding_window_size,
        std::nullopt,  // chunk_start_idx
        std::nullopt,  // chunk_start_idx_tensor
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        cu_window_seqlens,
        windowed_q_token_offset,
        windowed_q_token_offset_tensor);
}

// Legacy: chunk_start_idx as scalar (part of program cache key).
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::nullopt,        // attn_mask
        page_table_tensor,   // page_table
        std::nullopt,        // attention_sink
        /*is_causal=*/true,  // Always causal for chunked version
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        chunk_start_idx,
        std::nullopt,  // chunk_start_idx_tensor
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        std::nullopt,  // cu_window_seqlens
        0,             // windowed_q_token_offset (windowed mode only)
        std::nullopt,  // windowed_q_token_offset_tensor
        paged_cache_geometry);
}

// Flexible: chunk_start_idx in device tensor [1]; read at runtime (for tracing).
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& chunk_start_idx_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::nullopt,       // attn_mask
        page_table_tensor,  // page_table
        std::nullopt,       // attention_sink
        /*is_causal=*/true,
        scale,
        std::nullopt,  // sliding_window_size
        std::nullopt,
        chunk_start_idx_tensor,
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        std::nullopt,  // cu_window_seqlens
        0,             // windowed_q_token_offset (windowed mode only)
        std::nullopt,  // windowed_q_token_offset_tensor
        paged_cache_geometry);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<SDPAPrecision> precision,
    bool inputs_prepared) {
    if (precision) {
        namespace numeric = operations::transformer::sdpa::detail;
        TT_FATAL(joint_strategy == "rear", "SDPA recipes require rear joint strategy");
        const auto policy = numeric::resolve_recipe_policy(
            input_tensor_q, input_tensor_k, *precision, inputs_prepared, scale, compute_kernel_config, program_config);
        return numeric::run_joint_recipe(
            input_tensor_q,
            input_tensor_k,
            input_tensor_v,
            joint_tensor_q,
            joint_tensor_k,
            joint_tensor_v,
            policy,
            program_config);
    }
    TT_FATAL(!inputs_prepared, "inputs_prepared requires an explicit LOW_PRECISION recipe");
    auto output_tensors = ttnn::prim::joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        joint_strategy,
        program_config,
        scale,
        compute_kernel_config);
    return {output_tensors[prim::JOINT_SDPA_OUTPUT_IDX], output_tensors[prim::JOINT_SDPA_JOINT_OUTPUT_IDX]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    const LogicalLength& logical_n,
    const LogicalLength& logical_l,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const CoreCoord ccl_core_grid_offset,
    bool is_causal,
    bool is_balanced,
    bool is_cross,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& attention_sink,
    std::optional<uint32_t> sliding_window_size,
    bool circular_kv_cache,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_k,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_v,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor,
    std::optional<uint32_t> kv_cache_num_layers,
    std::optional<uint32_t> kv_cache_layer_idx,
    std::optional<SDPAPrecision> precision,
    bool inputs_prepared) {
    if (precision) {
        const auto policy = operations::transformer::sdpa::detail::resolve_recipe_policy(
            input_tensor_q, input_tensor_k, *precision, inputs_prepared, scale, compute_kernel_config, program_config);
        TT_FATAL(
            !is_causal && !is_balanced && !attention_sink && !sliding_window_size && !circular_kv_cache &&
                !kv_cache_batch_idx && !kv_actual_isl && !slot_id && !kv_actual_isl_tensor,
            "Named ring recipes currently require noncausal attention without indexed/cache/window/sink features");
        TT_FATAL(
            std::holds_alternative<std::size_t>(logical_n) && std::holds_alternative<std::size_t>(logical_l),
            "Named ring recipes currently require scalar logical lengths");
        TT_FATAL(
            input_tensor_q.logical_shape()[3] == 128 && input_tensor_k.logical_shape()[3] == 128 &&
                input_tensor_v.logical_shape()[3] == 128,
            "Named ring recipes require D128");
        operations::transformer::sdpa::detail::recipe_q_tiles(program_config);
        operations::transformer::sdpa::detail::recipe_k_tiles(program_config);
        TT_FATAL(
            input_tensor_q.dtype() == DataType::BFLOAT16 && input_tensor_k.dtype() == input_tensor_v.dtype(),
            "Named ring recipes require BF16 Q and matching KV types");
        TT_FATAL(
            is_cross || input_tensor_q.logical_shape()[2] == input_tensor_k.logical_shape()[2],
            "Named ring recipes do not yet support chunked prefill; use is_cross for noncausal cross attention");
        compute_kernel_config = BlackholeComputeKernelConfig{
            .math_fidelity = policy.qk_fidelity,
            .math_approx_mode = true,
            .fp32_dest_acc_en = policy.fp32_destination,
        };
    } else {
        TT_FATAL(!inputs_prepared, "inputs_prepared requires an explicit LOW_PRECISION recipe");
    }
    // Normalize empty joints to nullopt (see drop_if_empty).
    const std::optional<ttnn::Tensor> joint_q = drop_if_empty(joint_tensor_q);
    const std::optional<ttnn::Tensor> joint_k = drop_if_empty(joint_tensor_k);
    const std::optional<ttnn::Tensor> joint_v = drop_if_empty(joint_tensor_v);

    // Split each logical length into (scalar attribute, optional device tensor); on the tensor path the
    // attribute becomes the worst-case placeholder (see RingJointSDPAInputs).
    const std::size_t ring_size =
        (cluster_axis == 0) ? mesh_device.get_view().num_rows() : mesh_device.get_view().num_cols();
    const std::size_t padded_ring_n = static_cast<std::size_t>(input_tensor_k.logical_shape()[2]) * ring_size;
    const std::size_t padded_ring_l =
        joint_k.has_value() ? static_cast<std::size_t>(joint_k->logical_shape()[2]) * ring_size : 0;
    const auto split_logical_length =
        [](const LogicalLength& length,
           std::size_t placeholder) -> std::pair<std::size_t, std::optional<ttnn::Tensor>> {
        if (const auto* scalar = std::get_if<std::size_t>(&length)) {
            return {*scalar, std::nullopt};
        }
        return {placeholder, std::get<ttnn::Tensor>(length)};
    };
    const auto [logical_n_scalar, logical_n_tensor] = split_logical_length(logical_n, padded_ring_n);
    const auto [logical_l_scalar, logical_l_tensor] = split_logical_length(logical_l, padded_ring_l);

    auto topology_1d = ttnn::ccl::convert_2d_to_1d_topology(topology);
    auto output_tensors = ttnn::prim::ring_joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,  // AllGather input
        input_tensor_v,  // AllGather input
        joint_q,
        joint_k,
        joint_v,
        persistent_output_buffer_k,  // AllGather output / RingAttention input
        persistent_output_buffer_v,  // AllGather output / RingAttention input
        persistent_output_buffer_joint_k,
        persistent_output_buffer_joint_v,
        joint_strategy,
        logical_n_scalar,
        logical_l_scalar,
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology_1d,
        ccl_core_grid_offset,
        subdevice_id,
        is_causal,
        is_balanced,
        is_cross,
        scale,
        compute_kernel_config,
        core_allocation_strategy,
        kv_cache_batch_idx,
        kv_actual_isl,
        std::nullopt,  // latent_v_head_dim
        attention_sink,
        slot_id,
        kv_actual_isl_tensor,
        // Resolve to (1, 0) when unset so the readers compute slot = slot_id[0], the
        // pre-existing behaviour for callers that pass no layer packing.
        kv_cache_num_layers.value_or(1),
        kv_cache_layer_idx.value_or(0),
        sliding_window_size,
        circular_kv_cache,
        logical_n_tensor,
        logical_l_tensor,
        precision);
    return {
        output_tensors[prim::RING_JOINT_SDPA_OUTPUT_IDX],
        output_tensors[prim::RING_JOINT_SDPA_JOINT_OUTPUT_IDX],
        output_tensors[prim::RING_JOINT_SDPA_STATS_OUTPUT_IDX]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_mla(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_kv,
    ttnn::Tensor& persistent_output_buffer_kv,
    const uint32_t head_dim_v,
    std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<uint32_t> cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const CoreCoord ccl_core_grid_offset,
    bool is_balanced,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor,
    std::optional<uint32_t> kv_cache_num_layers,
    std::optional<uint32_t> kv_cache_layer_idx) {
    auto output_tensors = ttnn::prim::ring_joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_kv,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        persistent_output_buffer_kv,
        std::nullopt,  // persistent_output_buffer_v
        std::nullopt,  // persistent_output_buffer_joint_k
        std::nullopt,  // persistent_output_buffer_joint_v
        "rear",
        logical_n,
        /*logical_l=*/static_cast<std::size_t>(0),
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        ccl_core_grid_offset,
        subdevice_id,
        /*is_causal=*/true,
        is_balanced,
        /*is_cross=*/false,
        scale,
        compute_kernel_config,
        core_allocation_strategy,
        kv_cache_batch_idx,
        kv_actual_isl,
        head_dim_v,
        std::nullopt,  // attention_sink
        slot_id,
        kv_actual_isl_tensor,
        kv_cache_num_layers.value_or(1),
        kv_cache_layer_idx.value_or(0),
        std::nullopt);  // sliding_window_size
    return {output_tensors[prim::RING_JOINT_SDPA_OUTPUT_IDX], output_tensors[prim::RING_JOINT_SDPA_STATS_OUTPUT_IDX]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ExecuteExpRingJointAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    const LogicalLength& logical_n,
    operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const uint32_t num_workers_per_link,
    const uint32_t num_buffers_per_channel,
    std::optional<SDPAPrecision> precision,
    bool inputs_prepared) {
    if (precision) {
        // resolve_recipe_policy rejects an explicit compute_kernel_config, exp_approx_mode=False, a
        // non-default scale and a preparation mismatch; the recipe owns those numerical decisions.
        const auto policy = operations::transformer::sdpa::detail::resolve_recipe_policy(
            input_tensor_q, input_tensor_k, *precision, inputs_prepared, scale, compute_kernel_config, program_config);
        TT_FATAL(
            std::holds_alternative<std::size_t>(logical_n),
            "Named exp ring recipes currently require a scalar logical_n");
        TT_FATAL(
            program_config.k_chunk_size == 512 && input_tensor_q.logical_shape()[3] == 128 &&
                input_tensor_k.logical_shape()[3] == 128 && input_tensor_v.logical_shape()[3] == 128,
            "Named exp ring recipes require K512/D128");
        operations::transformer::sdpa::detail::recipe_q_tiles(program_config);
        TT_FATAL(
            input_tensor_q.dtype() == DataType::BFLOAT16 && input_tensor_k.dtype() == input_tensor_v.dtype(),
            "Named exp ring recipes require BF16 Q and matching KV types");
        compute_kernel_config = BlackholeComputeKernelConfig{
            .math_fidelity = policy.qk_fidelity,
            .math_approx_mode = true,
            .fp32_dest_acc_en = policy.fp32_destination,
        };
    } else {
        TT_FATAL(!inputs_prepared, "inputs_prepared requires an explicit LOW_PRECISION recipe");
    }
    // Normalize empty joints to nullopt (see drop_if_empty).
    const std::optional<ttnn::Tensor> joint_q = drop_if_empty(joint_tensor_q);
    const std::optional<ttnn::Tensor> joint_k = drop_if_empty(joint_tensor_k);
    const std::optional<ttnn::Tensor> joint_v = drop_if_empty(joint_tensor_v);

    // Tensor path: the scalar attribute becomes the worst-case placeholder (see ExpRingJointSDPAInputs).
    const std::size_t ring_size =
        (cluster_axis == 0) ? mesh_device.get_view().num_rows() : mesh_device.get_view().num_cols();
    const std::size_t padded_ring_n = static_cast<std::size_t>(input_tensor_k.logical_shape()[2]) * ring_size;
    std::size_t logical_n_scalar = padded_ring_n;
    std::optional<ttnn::Tensor> logical_n_tensor;
    if (const auto* scalar = std::get_if<std::size_t>(&logical_n)) {
        logical_n_scalar = *scalar;
    } else {
        logical_n_tensor = std::get<ttnn::Tensor>(logical_n);
    }

    auto output_tensors = ttnn::prim::exp_ring_joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,  // AllGather input
        input_tensor_v,  // AllGather input
        joint_q,
        joint_k,
        joint_v,
        persistent_output_buffer_k,  // AllGather output / RingAttention input
        persistent_output_buffer_v,  // AllGather output / RingAttention input
        joint_strategy,
        logical_n_scalar,
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        subdevice_id,
        scale,
        compute_kernel_config,
        num_workers_per_link,
        num_buffers_per_channel,
        logical_n_tensor,
        precision);
    return {
        output_tensors[prim::EXP_RING_JOINT_SDPA_OUTPUT_IDX],
        output_tensors[prim::EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX],
        output_tensors[prim::EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX]};
}

ttnn::Tensor flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const std::optional<ttnn::Tensor>& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask,
        std::nullopt,  // page_table
        std::nullopt,  // attention_sink
        is_causal,
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        std::nullopt,  // chunk_start_idx
        std::nullopt,  // chunk_start_idx_tensor
        true,          // use_mla
        head_dim_v,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

ttnn::Tensor chunked_flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        std::nullopt,       // V is implied by K in MLA mode
        std::nullopt,       // attn_mask
        page_table_tensor,  // page_table
        std::nullopt,       // attention_sink
        /*is_causal=*/true,
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        chunk_start_idx,
        std::nullopt,  // chunk_start_idx_tensor
        true,          // use_mla
        head_dim_v,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

ttnn::Tensor ring_distributed_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    uint32_t ring_size,
    std::optional<uint32_t>
        ring_id,  // Optional: if provided, uses this value; if nullopt, infers from device coordinate
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& page_table,
    std::optional<int64_t> chunk_start_idx) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::ring_distributed_sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        ring_size,
        ring_id,  // Pass through the ring_id parameter (can be used or ignored)
        scale,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        kernel_config_val,
        page_table,
        chunk_start_idx);
}

}  // namespace ttnn::transformer
