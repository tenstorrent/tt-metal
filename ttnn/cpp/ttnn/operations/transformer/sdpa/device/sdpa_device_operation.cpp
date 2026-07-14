// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_perf_model.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

void SDPAOperation::validate_on_program_cache_miss(const SDPAParams& attrs, const SDPAInputs& tensors) {
    const bool use_mla = attrs.use_mla;

    // Common validations for both modes
    if (use_mla) {
        TT_FATAL(attrs.head_dim_v.has_value(), "Must provide head_dim_v for multi-latent attention decode");
    } else {
        TT_FATAL(
            tensors.v.has_value(),
            "Non-MLA SDPA requires V tensor to be provided (Q, K, V). V tensor not provided in tensor_args.");
    }

    const Tensor& q = tensors.q;
    const Tensor& k = tensors.k;
    const Tensor& v = use_mla ? tensors.k : tensors.v.value();
    const auto input_tile = q.tensor_spec().tile();
    const auto input_tile_width = input_tile.get_width();
    const auto input_tile_hw = input_tile.get_tile_hw();
    // Basic tensor properties
    for (const auto* input_tensor : {&q, &k, &v}) {
        TT_FATAL(input_tensor->storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device");
        TT_FATAL(input_tensor->buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device");
        TT_FATAL((input_tensor->layout() == Layout::TILE), "Inputs to SDPA must be tilized");
        TT_FATAL(
            input_tensor->dtype() == DataType::BFLOAT16 || input_tensor->dtype() == DataType::BFLOAT8_B ||
                input_tensor->dtype() == DataType::BFLOAT4_B,
            "Data type of input tensor must be BFLOAT16, BFLOAT8_B, or BFLOAT4_B and is {}",
            input_tensor->dtype());
        TT_FATAL(!input_tensor->is_sharded(), "Operands to SDPA need to be DRAM/L1 interleaved");
        TT_FATAL(
            input_tensor->tensor_spec().tile() == input_tile,
            "Inputs to SDPA must have the same tile size, expected {}, got {}",
            input_tile,
            input_tensor->tensor_spec().tile());
    }

    auto validate_padding = [&](const Tensor& tensor) {
        auto logical_shape = tensor.logical_shape();
        auto legacy_shape = tensor.padded_shape();
        TT_FATAL(logical_shape[0] == legacy_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == legacy_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == legacy_shape[3], "Padding is not supported on the head_dim dimension");
    };

    // Q/K/V shape agreement + chunk-size checks shared by regular and windowed modes.
    auto validate_shapes_and_chunks = [&]() {
        const auto q_shape = q.logical_shape();
        const auto k_shape = k.logical_shape();
        const auto v_shape = v.logical_shape();
        const auto B = q_shape[0];
        const auto nqh = q_shape[1];
        const auto nkv = k_shape[1];
        const auto Sq = q_shape[2];
        const auto DH = q_shape[3];
        const auto Sk = k_shape[2];
        if (attrs.is_causal) {
            TT_FATAL(
                Sq == Sk, "Causal SDPA requires Q and K to have the same sequence length. Got Q: {}, K: {}", Sq, Sk);
        }

        if (use_mla) {
            // Head dim v validation
            TT_FATAL(
                attrs.head_dim_v.value() <= q_shape[3],
                "Head dimension of V must be less than or equal to head dim of Q, got {} and {}",
                attrs.head_dim_v.value(),
                q_shape[3]);
            TT_FATAL(
                attrs.head_dim_v.value() <= k_shape[3],
                "Head dimension of V must be less than or equal to head dim of K, got {} and {}",
                attrs.head_dim_v.value(),
                q_shape[3]);
        } else {
            TT_FATAL(
                k_shape[0] == B && v_shape[0] == B,
                "K and V batch must match. Got K: {}, V: {}",
                k_shape[0],
                v_shape[0]);
            TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
            TT_FATAL(v_shape[2] == Sk, "K and V sequence length must match. Got K: {}, V: {}", k_shape[2], v_shape[2]);
            TT_FATAL(
                k_shape[3] == DH && v_shape[3] == DH,
                "K and V hidden dim must match. Got K: {}, V: {}",
                k_shape[3],
                v_shape[3]);
        }
        TT_FATAL(
            nqh >= nkv && nqh % nkv == 0,
            "Q num_heads must be >= K num_heads and divisible by K num_heads. Got Q: {}, K: {}",
            nqh,
            nkv);

        if (attrs.program_config.has_value()) {
            auto q_chunk_size = attrs.program_config->q_chunk_size;
            auto k_chunk_size = attrs.program_config->k_chunk_size;

            TT_FATAL(
                q_chunk_size % input_tile_width == 0,
                "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
                q_chunk_size,
                input_tile_width);
            TT_FATAL(
                k_chunk_size % input_tile_width == 0,
                "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
                k_chunk_size,
                input_tile_width);
        }
    };

    auto validate_regular_mode = [&]() {
        TT_FATAL(
            !(attrs.is_causal && tensors.attn_mask.has_value()),
            "is_causal and attn_mask cannot both be present. Got is_causal: {}, attn_mask: {}",
            attrs.is_causal,
            tensors.attn_mask.has_value());

        // A user-provided dense mask runs on the streaming compute kernel, which applies the mask
        // and the structured sliding-window stamp through the same L1-accumulate slot and treats
        // them as mutually exclusive (static_assert in sdpa_standard_v2). Sliding-window masking is
        // expected to be baked into the provided mask instead. Reject the combination here so the
        // caller gets a clear error rather than a kernel build failure.
        TT_FATAL(
            !(attrs.sliding_window_size.value_or(0) > 0 && tensors.attn_mask.has_value()),
            "sliding_window_size and attn_mask cannot both be present; bake the sliding-window mask "
            "into attn_mask. Got sliding_window_size: {}, attn_mask: {}",
            attrs.sliding_window_size.value_or(0),
            tensors.attn_mask.has_value());

        const auto& mask_option = tensors.attn_mask;
        if (mask_option.has_value()) {
            const auto& mask = mask_option.value();
            TT_FATAL(
                mask.storage_type() == StorageType::DEVICE,
                "When mask is provided to SDPA, the tensor must be on device");
            TT_FATAL(
                q.device() == mask.device(),
                "When mask is provided to SDPA, it must be on the same device as the input tensors");
            TT_FATAL(mask.layout() == Layout::TILE, "When mask is provided to SDPA, it must be tilized");
            TT_FATAL(
                mask.dtype() == DataType::BFLOAT16 || mask.dtype() == DataType::BFLOAT8_B ||
                    mask.dtype() == DataType::BFLOAT4_B,
                "When mask is provided to SDPA, it must be in BF16, BFP8, or BFP4 dataformat");

            TT_FATAL(
                mask.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
                "When mask is provided to SDPA, it must be in DRAM");

            const auto& mask_shape = mask.logical_shape();
            const auto q_shape = q.logical_shape();
            const auto k_shape = k.logical_shape();

            TT_FATAL(
                mask_shape[0] == 1 || mask_shape[0] == q_shape[0],
                "Mask batch dim must either be 1 (to be broadcasted across all batches) or must match Q batch "
                "dimension");
            TT_FATAL(
                mask_shape[1] == 1 || mask_shape[1] == q_shape[1],
                "Mask num_heads must either be 1 (to be broadcasted across all heads) or must match Q heads dimension");
            TT_FATAL(mask_shape[2] == q_shape[2], "Mask sequence length must match Q sequence length");
            TT_FATAL(mask_shape[3] == k_shape[2], "Mask sequence length must match K sequence length");
        }

        validate_shapes_and_chunks();
    };

    auto validate_chunked_mode = [&]() {
        const bool has_tensor_chunk_start_idx = attrs.chunk_start_idx_tensor.has_value();
        const bool has_scalar_chunk_start_idx = attrs.chunk_start_idx.has_value();

        // Disallow ambiguous configuration where both scalar and tensor start indices are provided.
        TT_FATAL(
            !(has_scalar_chunk_start_idx && has_tensor_chunk_start_idx),
            "chunked mode accepts only one of chunk_start_idx (scalar) or chunk_start_idx_tensor; both were provided");

        const bool flexible_chunked = has_tensor_chunk_start_idx;
        const bool legacy_chunked = has_scalar_chunk_start_idx && !has_tensor_chunk_start_idx;
        TT_FATAL(
            legacy_chunked || flexible_chunked,
            "chunked mode requires either chunk_start_idx (scalar) or chunk_start_idx_tensor");
        if (legacy_chunked) {
            TT_FATAL(attrs.chunk_start_idx.value() >= 0, "chunk_start_idx must be non-negative");
        }
        if (flexible_chunked) {
            const auto& csi_tensor = attrs.chunk_start_idx_tensor.value();
            TT_FATAL(
                tensors.chunk_start_idx_tensor.has_value(),
                "chunk_start_idx tensor must be present in tensor args for descriptor cache patching");
            TT_FATAL(
                tensors.chunk_start_idx_tensor.value().buffer() == csi_tensor.buffer(),
                "chunk_start_idx tensor in attrs and tensor args must reference the same buffer");
            TT_FATAL(csi_tensor.storage_type() == StorageType::DEVICE, "chunk_start_idx tensor must be on device");
            TT_FATAL(
                csi_tensor.dtype() == DataType::INT32,
                "chunk_start_idx tensor must be int32. Got {}",
                csi_tensor.dtype());
            auto csi_shape = csi_tensor.logical_shape();
            TT_FATAL(
                csi_shape.size() == 1 && csi_shape[0] == 1,
                "chunk_start_idx tensor must have shape [1]. Got {}",
                csi_shape);
        }
        // Validate page table tensor
        const auto& page_table = tensors.page_table.value();
        TT_FATAL(page_table.storage_type() == StorageType::DEVICE, "Page table tensor must be on device");
        TT_FATAL(q.device() == page_table.device(), "Page table must be on the same device as the input tensors");
        TT_FATAL(page_table.layout() == Layout::ROW_MAJOR, "Page table must be row major");
        // Check that page table is int32
        TT_FATAL(page_table.dtype() == DataType::INT32, "Page table must be int32");
        // Validate that first optional tensor (mask) is not provided
        TT_FATAL(
            !tensors.attn_mask.has_value(),
            "Attention mask should not be provided in chunked mode - masking is handled internally");

        // Additional chunked-specific validations
        const auto q_shape = q.logical_shape();
        const auto k_shape = k.logical_shape();
        const auto v_shape = v.logical_shape();
        const auto page_table_shape = page_table.logical_shape();
        const auto B = q_shape[0];
        const auto nqh = q_shape[1];
        const auto nkv = k_shape[1];
        const auto DH = q_shape[3];
        const auto k_page_size = k_shape[2];
        const uint32_t num_pages_per_user = page_table.logical_shape()[1];
        if (!use_mla) {
            // Check that k page size matches v page size
            TT_FATAL(
                k_page_size == v_shape[2],
                "K page size must match V page size. Got K: {}, V: {}",
                k_page_size,
                v_shape[2]);
        }
        // Check that page table has same batch size as input tensors
        TT_FATAL(
            page_table_shape[0] == B,
            "Page table batch size must match input batch size. Got Page table: {}, Input: {}",
            page_table_shape[0],
            B);
        // Calculate K length based on number of pages per user
        const uint32_t kv_length = num_pages_per_user * k_page_size;

        if (!use_mla) {
            TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
            TT_FATAL(
                k_shape[3] == DH && v_shape[3] == DH,
                "K and V hidden dim must match. Got K: {}, V: {}",
                k_shape[3],
                v_shape[3]);
        }
        TT_FATAL(
            nqh >= nkv && nqh % nkv == 0,
            "Q num_heads must be >= K num_heads and divisible by K num_heads. Got Q: {}, K: {}",
            nqh,
            nkv);

        if (attrs.program_config.has_value()) {
            auto q_chunk_size = attrs.program_config->q_chunk_size;
            auto k_chunk_size = attrs.program_config->k_chunk_size;

            TT_FATAL(
                q_chunk_size % input_tile_width == 0,
                "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
                q_chunk_size,
                input_tile_width);
            TT_FATAL(
                k_chunk_size % input_tile_width == 0,
                "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
                k_chunk_size,
                input_tile_width);

            if (legacy_chunked) {
                // Validate that chunk_start_idx is a multiple of q_chunk_size
                // This is required because chunk_start_idx is divided by q_chunk_size to compute chunked_q_chunk_offset
                TT_FATAL(
                    attrs.chunk_start_idx.value() % q_chunk_size == 0,
                    "chunk_start_idx must be a multiple of q_chunk_size. Got chunk_start_idx: {}, q_chunk_size: {}",
                    attrs.chunk_start_idx.value(),
                    q_chunk_size);

                // Validate that chunk_start_idx is a multiple of k_chunk_size
                // Workaround for https://github.com/tenstorrent/tt-metal/issues/35225
                TT_FATAL(
                    attrs.chunk_start_idx.value() % k_chunk_size == 0,
                    "chunk_start_idx must be a multiple of k_chunk_size. Got chunk_start_idx: {}, k_chunk_size: {}",
                    attrs.chunk_start_idx.value(),
                    k_chunk_size);
            }
        }

        if (legacy_chunked) {
            // In chunked mode, K's sequence dimension should be >= Q's sequence dimension + chunk_start_idx
            TT_FATAL(
                kv_length >= q_shape[2] + attrs.chunk_start_idx.value(),
                "K's sequence length must be >= Q's sequence length + chunk_start_idx. Got K: {}, Q: {}, "
                "chunk_start_idx: "
                "{}",
                kv_length,
                q_shape[2],
                attrs.chunk_start_idx.value());
        } else {
            // Flexible: only require KV length to cover Q
            TT_FATAL(
                kv_length >= q_shape[2],
                "K's sequence length must be >= Q's sequence length. Got K: {}, Q: {}",
                kv_length,
                q_shape[2]);
        }
    };

    auto validate_attention_sink = [&]() {
        // Validate attention sink if provided (optional_input_tensors[2])
        if (tensors.attention_sink.has_value()) {
            const auto& attention_sink = tensors.attention_sink.value();
            TT_FATAL(attention_sink.storage_type() == StorageType::DEVICE, "Attention sink tensor must be on device");
            TT_FATAL(
                q.device() == attention_sink.device(),
                "Attention sink must be on the same device as the input tensors");
            TT_FATAL(attention_sink.layout() == Layout::TILE, "Attention sink must be tilized");
            TT_FATAL(
                attention_sink.dtype() == DataType::BFLOAT16 || attention_sink.dtype() == DataType::BFLOAT8_B ||
                    attention_sink.dtype() == DataType::BFLOAT4_B,
                "Attention sink must be in BF16, BFP8, or BFP4 dataformat");
            TT_FATAL(
                attention_sink.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
                "Attention sink must be in DRAM");

            const auto& sink_shape = attention_sink.logical_shape();
            const auto q_shape = q.logical_shape();

            // Attention sink must have shape [1, NH, 1, 1] - single value per head, broadcast across batch and tile
            // dims
            TT_FATAL(sink_shape[0] == 1, "Attention sink batch dimension must be 1. Got {}", sink_shape[0]);
            TT_FATAL(
                sink_shape[1] == q_shape[1],
                "Attention sink num_heads must match Q num_heads. Got sink: {}, Q: {}",
                sink_shape[1],
                q_shape[1]);
            TT_FATAL(sink_shape[2] == 1, "Attention sink sequence dimension must be 1. Got {}", sink_shape[2]);
            TT_FATAL(sink_shape[3] == 1, "Attention sink hidden dimension must be 1. Got {}", sink_shape[3]);
        }
    };

    auto check_conditions = [&]() {
        bool has_chunk_start_scalar = attrs.chunk_start_idx.has_value();
        bool has_chunk_start_tensor = attrs.chunk_start_idx_tensor.has_value();
        bool has_page_table = tensors.page_table.has_value();
        if (has_chunk_start_scalar || has_chunk_start_tensor) {
            TT_FATAL(has_page_table, "page_table must be provided for chunked mode");
        }
    };

    auto validate_windowed_mode = [&]() {
        TT_FATAL(tensors.cu_window_seqlens.has_value(), "Windowed SDPA requires cu_window_seqlens.");
        TT_FATAL(!attrs.is_causal, "Windowed SDPA is non-causal; is_causal must be false.");
        TT_FATAL(
            !tensors.attn_mask.has_value(),
            "Windowed SDPA builds its mask from cu_window_seqlens; attn_mask must not be provided.");
        TT_FATAL(!attrs.use_mla, "Windowed SDPA does not support MLA.");
        TT_FATAL(
            !(attrs.chunk_start_idx.has_value() || attrs.chunk_start_idx_tensor.has_value()),
            "Windowed SDPA does not support chunked/paged mode.");
        TT_FATAL(attrs.sliding_window_size.value_or(0) == 0, "Windowed SDPA does not support sliding_window_size.");
        TT_FATAL(!tensors.attention_sink.has_value(), "Windowed SDPA does not support attention_sink.");

        // Windowed attention is otherwise plain non-causal SDPA, so apply the same Q/K/V shape and
        // chunk-size validation as the regular path.
        validate_shapes_and_chunks();

        const auto& cu = tensors.cu_window_seqlens.value();
        TT_FATAL(cu.storage_type() == StorageType::DEVICE, "cu_window_seqlens must be on device.");
        TT_FATAL(cu.buffer() != nullptr, "cu_window_seqlens must be allocated on device.");
        TT_FATAL(q.device() == cu.device(), "cu_window_seqlens must be on the same device as Q/K/V.");
        TT_FATAL(
            cu.dtype() == DataType::INT32 || cu.dtype() == DataType::UINT32,
            "cu_window_seqlens must be INT32/UINT32, got {}.",
            cu.dtype());
        TT_FATAL(cu.layout() == Layout::ROW_MAJOR, "cu_window_seqlens must be ROW_MAJOR.");
        // Must be a 1-D tensor of cumulative boundaries [0, w1, ..., S]: the writer reads it as a flat
        // array and indexes up to (num_elements - 1), so at least two entries are required.
        const auto& cu_shape = cu.logical_shape();
        TT_FATAL(cu_shape.rank() == 1, "cu_window_seqlens must be 1-D, got rank {}.", cu_shape.rank());
        const auto cu_eles = cu_shape[-1];
        // The writer loads cu_window_seqlens into a single CB tile and the generator indexes the whole
        // array from it, so the element count is bounded by one uint32 tile (TILE_HW entries). Supporting
        // more would require a multi-tile load in writer_interleaved.cpp / windowed_mask_gen.hpp.
        const uint32_t max_cu_window_seqlens = input_tile_hw;  // 1024 uint32 per tile
        TT_FATAL(
            cu_eles >= 2 && cu_eles <= max_cu_window_seqlens,
            "cu_window_seqlens must have between 2 and {} elements, got {}.",
            max_cu_window_seqlens,
            cu_eles);
    };

    check_conditions();
    bool is_chunked_mode = attrs.chunk_start_idx.has_value() || attrs.chunk_start_idx_tensor.has_value();

    if (attrs.is_windowed) {
        validate_windowed_mode();
    } else if (is_chunked_mode) {
        validate_chunked_mode();
    } else {
        validate_regular_mode();
    }

    // Validate attention sink if provided
    validate_attention_sink();

    // Check padding: Only the sequence dimension may be padded. For all other dims, logical shape must be equal to
    // legacy shape
    for (const auto* tensor : {&q, &k, &v}) {
        validate_padding(*tensor);
    }
}

SDPAOperation::spec_return_value_t SDPAOperation::compute_output_specs(
    const SDPAParams& attrs, const SDPAInputs& tensors) {
    auto shape = tensors.q.logical_shape();
    if (attrs.use_mla) {
        shape[3] = attrs.head_dim_v.value_or(shape[3]);
    }
    return TensorSpec(
        shape,
        TensorLayout(
            tensors.q.dtype(), PageConfig(Layout::TILE, tensors.q.tensor_spec().tile()), attrs.output_mem_config));
}

SDPAOperation::tensor_return_value_t SDPAOperation::create_output_tensors(
    const SDPAParams& attrs, const SDPAInputs& tensors) {
    return create_device_tensor(compute_output_specs(attrs, tensors), tensors.q.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SDPAOperation::tensor_return_value_t>
SDPAOperation::create_op_performance_model(
    const SDPAParams& args, const SDPAInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const bool has_v = tensor_args.v.has_value();
    const auto& input_tensor_v = tensor_args.v.value_or(tensor_args.k);

    if (output_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    // Build input tensors list
    Tensors input_tensors = {tensor_args.q, tensor_args.k};
    if (tensor_args.v.has_value()) {
        input_tensors.emplace_back(tensor_args.v.value());
    }

    // Calculate arch specific parameters
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "SDPA perf model does not support tt::arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<SDPAOperation::tensor_return_value_t>(
            input_tensors, output_tensor, 0);
    }

    // Get main dimensions for Q*K and softmax(QK^T/sqrt) * V matmuls
    auto q_shape = input_tensor_q.logical_shape();
    auto k_shape = input_tensor_k.logical_shape();
    auto v_shape = input_tensor_v.logical_shape();
    TT_ASSERT(q_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor Q rank != 4");
    TT_ASSERT(k_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor K rank != 4");
    if (!args.use_mla) {
        TT_ASSERT(v_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor V rank != 4");
    }

    bool is_chunked_prefill = args.chunk_start_idx.has_value() || args.chunk_start_idx_tensor.has_value();

    uint32_t batch_size = q_shape[0];
    uint32_t num_heads_q = q_shape[1];
    const uint32_t Sq = q_shape[2];
    const uint32_t DH = q_shape[3];

    // Compute Sk based on chunked mode
    const uint32_t Sk = (is_chunked_prefill)
                            ? (args.chunk_start_idx.has_value() ? q_shape[2] + args.chunk_start_idx.value()
                                                                : k_shape[2])  // flexible: use K length as upper bound
                            : k_shape[2];

    // Compute DV based on MLA mode
    // Note: For MLA without V, use K's head dimension; otherwise use V's head dimension
    const uint32_t DV = (args.use_mla && !has_v) ? k_shape[3] : v_shape[3];

    TT_ASSERT(q_shape[0] == k_shape[0], "ScaledDotProductAttention perf model: Q and K have unequal batch size!");
    TT_ASSERT(q_shape[3] == k_shape[3], "ScaledDotProductAttention perf model: Q and K have unequal hidden dim!");

    CoreCoord compute_grid_dims = args.program_config.has_value()
                                      ? args.program_config->compute_with_storage_grid_size
                                      : output_tensor.device()->compute_with_storage_grid_size();
    tt::tt_metal::MathFidelity math_fidelity = ttnn::get_math_fidelity(args.compute_kernel_config);

    int ideal_dev_clock_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        batch_size,
        num_heads_q,
        Sq,
        Sk,
        DH,
        DV,
        args.is_causal,
        math_fidelity,
        compute_grid_dims.x * compute_grid_dims.y);

    // TODO: somehow account for overhead of fused masking and softmax?

    return operation::OpPerformanceModelGeneral<SDPAOperation::tensor_return_value_t>(
        input_tensors, output_tensor, ideal_dev_clock_cycles);
}

}  // namespace ttnn::prim

namespace ttnn::prim {
Tensor sdpa(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<Tensor>& input_tensor_v,
    const std::optional<Tensor>& attn_mask,
    const std::optional<Tensor>& page_table_tensor,
    const std::optional<Tensor>& attention_sink,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    std::optional<int64_t> chunk_start_idx,
    const std::optional<Tensor>& chunk_start_idx_tensor,
    bool use_mla,
    std::optional<uint32_t> head_dim_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& cu_window_seqlens) {
    using OperationType = ttnn::prim::SDPAOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .output_mem_config = output_mem_config,
            .program_config = std::move(program_config),
            .is_causal = is_causal,
            .chunk_start_idx = chunk_start_idx,
            .chunk_start_idx_tensor = chunk_start_idx_tensor,
            .compute_kernel_config = compute_kernel_config,
            .use_mla = use_mla,
            .head_dim_v = head_dim_v,
            .sliding_window_size = sliding_window_size,
            .is_windowed = cu_window_seqlens.has_value(),
        },
        OperationType::tensor_args_t{
            .q = input_tensor_q,
            .k = input_tensor_k,
            .v = input_tensor_v,
            .attn_mask = attn_mask,
            .page_table = page_table_tensor,
            .chunk_start_idx_tensor = chunk_start_idx_tensor,
            .attention_sink = attention_sink,
            .cu_window_seqlens = cu_window_seqlens,
        });
}
}  // namespace ttnn::prim
