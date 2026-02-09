// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_program_factory.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::transformer::sdpa {

namespace {

std::uint32_t get_q_chunk_size(const operation_attributes_t& attrs) {
    return attrs.program_config ? attrs.program_config->q_chunk_size : 32;
}

std::uint32_t get_k_chunk_size(const operation_attributes_t& attrs) {
    return attrs.program_config ? attrs.program_config->k_chunk_size : 32;
}

}  // namespace

SDPAOperation::program_factory_t SDPAOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::SDPAProgramFactory{};
}
void SDPAOperation::validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    validate_on_program_cache_miss(attrs, tensors);
}

void SDPAOperation::validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensors) {
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
    }

    auto validate_padding = [&](const Tensor& tensor) {
        auto logical_shape = tensor.logical_shape();
        auto legacy_shape = tensor.padded_shape();
        TT_FATAL(logical_shape[0] == legacy_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == legacy_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == legacy_shape[3], "Padding is not supported on the head_dim dimension");
    };

    auto validate_regular_mode = [&]() {
        TT_FATAL(
            !(attrs.is_causal && tensors.attn_mask.has_value()),
            "is_causal and attn_mask cannot both be present. Got is_causal: {}, attn_mask: {}",
            attrs.is_causal,
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

            TT_FATAL(mask_shape[0] == q_shape[0], "Mask batch dim must match Q batch dim");
            TT_FATAL(
                mask_shape[1] == 1 || mask_shape[1] == q_shape[1],
                "Mask num_heads must either be 1 (to be broadcasted across all heads) or must match Q heads dimension");
            TT_FATAL(mask_shape[2] == q_shape[2], "Mask sequence length must match Q sequence length");
            TT_FATAL(mask_shape[3] == k_shape[2], "Mask sequence length must match K sequence length");

            // When given a mask, we must check that the mask can be divided by chunk size. Otherwise we'd need to pad
            // the mask.
            const auto q_chunk_size = get_q_chunk_size(attrs);
            const auto k_chunk_size = get_k_chunk_size(attrs);
            TT_FATAL(
                q_shape[2] % q_chunk_size == 0,
                "If mask is provided, Q sequence length must be divisible by q_chunk_size. Got q_seq_len: {}, "
                "q_chunk_size: {}",
                q_shape[2],
                q_chunk_size);
            TT_FATAL(
                k_shape[2] % k_chunk_size == 0,
                "If mask is provided, K sequence length must be divisible by k_chunk_size. Got k_seq_len: {}, "
                "k_chunk_size: {}",
                k_shape[2],
                k_chunk_size);
        }

        // Shape checks
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
                q_chunk_size % tt::constants::TILE_WIDTH == 0,
                "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
                q_chunk_size,
                tt::constants::TILE_WIDTH);
            TT_FATAL(
                k_chunk_size % tt::constants::TILE_WIDTH == 0,
                "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
                k_chunk_size,
                tt::constants::TILE_WIDTH);
        }
    };

    auto validate_chunked_mode = [&]() {
        TT_FATAL(attrs.chunk_start_idx.has_value(), "chunk_start_idx must be provided for chunked mode");
        TT_FATAL(attrs.chunk_start_idx.value() >= 0, "chunk_start_idx must be non-negative");
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
                q_chunk_size % tt::constants::TILE_WIDTH == 0,
                "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
                q_chunk_size,
                tt::constants::TILE_WIDTH);
            TT_FATAL(
                k_chunk_size % tt::constants::TILE_WIDTH == 0,
                "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
                k_chunk_size,
                tt::constants::TILE_WIDTH);

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

        // In chunked mode, K's sequence dimension should be >= Q's sequence dimension + chunk_start_idx
        TT_FATAL(
            kv_length >= q_shape[2] + attrs.chunk_start_idx.value(),
            "K's sequence length must be >= Q's sequence length + chunk_start_idx. Got K: {}, Q: {}, chunk_start_idx: "
            "{}",
            kv_length,
            q_shape[2],
            attrs.chunk_start_idx.value());
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
        bool has_chunk_start = attrs.chunk_start_idx.has_value();
        bool has_page_table = tensors.page_table.has_value();
        // For chunked mode, we need at least 2 optional inputs (mask placeholder and page_table)
        if (has_chunk_start) {
            TT_FATAL(has_page_table, "page_table must be provided when chunk_start_idx is set");
        }
    };

    check_conditions();
    bool is_chunked_mode = attrs.chunk_start_idx.has_value();

    if (is_chunked_mode) {
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

spec_return_value_t SDPAOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    auto shape = tensors.q.logical_shape();
    if (attrs.use_mla) {
        shape[3] = attrs.head_dim_v.value_or(shape[3]);
    }
    return TensorSpec(shape, TensorLayout(tensors.q.dtype(), PageConfig(Layout::TILE), attrs.output_mem_config));
}

tensor_return_value_t SDPAOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(attrs, tensors), tensors.q.device());
}

tt::stl::hash::hash_t SDPAOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    bool is_chunked_prefill = attrs.chunk_start_idx.has_value();

    const Tensor& q = tensors.q;
    const Tensor& k = tensors.k;
    const Tensor& v = attrs.use_mla ? tensors.k : tensors.v.value_or(tensors.k);

    operation::Hash hash = operation::hash_operation<SDPAOperation>(
        attrs.head_dim_v,
        attrs.scale,
        attrs.sliding_window_size,
        attrs.output_mem_config,
        attrs.program_config,
        attrs.is_causal,
        is_chunked_prefill,
        attrs.compute_kernel_config,
        q,
        k,
        v,
        tensors.attn_mask,
        tensors.page_table,
        tensors.attention_sink);
    return hash;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> SDPAOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const auto& input_tensor_v = args.use_mla ? tensor_args.k : tensor_args.v.value();

    if (output_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    // calculate arch specific parameters
    MathFidelity math_fidelity = ttnn::get_math_fidelity(args.compute_kernel_config);
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();
    Tensors input_tensors = {tensor_args.q, tensor_args.k};
    if (tensor_args.v.has_value()) {
        input_tensors.emplace_back(tensor_args.v.value());
    }
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "SDPA perf model does not support tt::arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<tensor_return_value_t>(input_tensors, output_tensor, 0);
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

    bool is_chunked_prefill = args.chunk_start_idx.has_value();

    uint32_t batch_size_q = q_shape[0];
    uint32_t batch_size_k = k_shape[0];
    // NB: number of Q heads determines the shape of first matmul; K is broadcast to match Q's shape
    uint32_t num_heads_q = q_shape[1];

    const auto Sq = q_shape[2];
    const auto Sk = (is_chunked_prefill) ? q_shape[-2] + args.chunk_start_idx.value() : k_shape[2];
    const auto DH = q_shape[3];

    uint32_t Sv, DV;
    if (args.use_mla) {
        Sv = k_shape[2];
        DV = k_shape[3];
    } else {
        Sv = v_shape[3];
        DV = v_shape[2];
    }

    TT_ASSERT(batch_size_q == batch_size_k, "ScaledDotProductAttention perf model: Q and K have unequal batch size!");
    TT_ASSERT(q_shape[3] == k_shape[3], "ScaledDotProductAttention perf model: Q and K have unequal hidden dim!");

    // Compute number of FLOPS for the two main matmuls
    constexpr int64_t FLOPS_PER_FMA = 2;  // each FMA is 2 FLOPS
    int64_t num_mul_adds = 0;
    // Q * K matmul for raw attention scores
    num_mul_adds += FLOPS_PER_FMA * DH * Sq * Sk * num_heads_q * batch_size_q;
    // attention scores * V matmul
    num_mul_adds += FLOPS_PER_FMA * DV * Sq * Sv * num_heads_q * batch_size_q;

    // if causal, only half of the FMAs are actually performed
    if (args.is_causal) {
        num_mul_adds /= 2;
    }

    CoreCoord compute_grid_dims = output_tensor.device()->compute_with_storage_grid_size();
    int num_cores = compute_grid_dims.x * compute_grid_dims.y;

    // wormhole and blackhole have identical matmul throughput per cycle
    const int tensix_mul_adds_per_cycle_lofi = 4096;

    // ideal total cycles is ESTIMATED_FLOPS / IDEAL_THROUGHPUT
    int ideal_dev_clock_cycles = std::ceil(
        ((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        (float)operation::OpPerformanceModel::fidelity_multiplier(math_fidelity));

    // TODO: somehow account for overhead of fused masking and softmax?

    operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        input_tensors, output_tensor, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::transformer::sdpa

namespace ttnn::prim {
ttnn::operations::transformer::sdpa::SDPAOperation::tensor_return_value_t sdpa(
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
    bool use_mla,
    std::optional<uint32_t> head_dim_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::operations::transformer::sdpa::SDPAOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .output_mem_config = output_mem_config,
            .program_config = std::move(program_config),
            .is_causal = is_causal,
            .chunk_start_idx = chunk_start_idx,
            .compute_kernel_config = compute_kernel_config,
            .use_mla = use_mla,
            .head_dim_v = head_dim_v,
            .sliding_window_size = sliding_window_size,
        },
        OperationType::tensor_args_t{
            .q = input_tensor_q,
            .k = input_tensor_k,
            .v = input_tensor_v,
            .attn_mask = attn_mask,
            .page_table = page_table_tensor,
            .attention_sink = attention_sink,
        });
}
}  // namespace ttnn::prim
