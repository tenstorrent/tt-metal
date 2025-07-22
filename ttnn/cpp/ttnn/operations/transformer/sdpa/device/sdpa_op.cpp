// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_op.hpp"

#include "sdpa_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <enchantum/enchantum.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

void ScaledDotProductAttention::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    bool use_mla = this->use_mla.value_or(false);
    // Common validations for both modes
    if (use_mla) {
        TT_FATAL(input_tensors.size() == 2, "Must have 2 input tensors (Q, K)");
        TT_FATAL(this->head_dim_v.has_value(), "Must provide head_dim_v for multi-latent attention decode");
    } else {
        TT_FATAL(input_tensors.size() == 3, "Must have 3 input tensors (Q, K, V)");
    }
    TT_FATAL(
        optional_input_tensors.size() == 1 or optional_input_tensors.size() == 2,
        "Must have 1 or 2 optional tensors (mask/page_table)");

    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device");
        TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to SDPA must be tilized");
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B ||
                input_tensor.dtype() == DataType::BFLOAT4_B,
            "Data type of input tensor must be BFLOAT16, BFLOAT8_B, or BFLOAT4_B and is {}",
            input_tensor.dtype());
        TT_FATAL(
            input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to SDPA need to be in DRAM");
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
            !(this->is_causal && optional_input_tensors.at(0).has_value()),
            "is_causal and attn_mask cannot both be present. Got is_causal: {}, attn_mask: {}",
            this->is_causal,
            optional_input_tensors.at(0).has_value());

        const auto& mask_option = optional_input_tensors.at(0);
        if (mask_option.has_value()) {
            const auto& mask = mask_option.value();
            TT_FATAL(
                mask.storage_type() == StorageType::DEVICE,
                "When mask is provided to SDPA, the tensor must be on device");
            TT_FATAL(
                input_tensors.at(0).device() == mask.device(),
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
            const auto q_shape = input_tensors.at(0).logical_shape();
            const auto k_shape = input_tensors.at(1).logical_shape();

            TT_FATAL(mask_shape[0] == q_shape[0], "Mask batch dim must match Q batch dim");
            TT_FATAL(mask_shape[1] == 1, "Mask num_heads must be 1 to be broadcasted across all heads");
            TT_FATAL(mask_shape[2] == q_shape[2], "Mask sequence length must match Q sequence length");
            TT_FATAL(mask_shape[3] == k_shape[2], "Mask sequence length must match K sequence length");

            // When given a mask, we must check that the mask can be divided by chunk size. Otherwise we'd need to pad
            // the mask.
            const auto q_chunk_size = this->get_q_chunk_size();
            const auto k_chunk_size = this->get_k_chunk_size();
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
        const auto q_shape = input_tensors.at(0).logical_shape();
        const auto k_shape = input_tensors.at(1).logical_shape();
        const auto v_shape = use_mla ? input_tensors.at(1).logical_shape() : input_tensors.at(2).logical_shape();
        const auto B = q_shape[0];
        const auto nqh = q_shape[1];
        const auto nkv = k_shape[1];
        const auto Sq = q_shape[2];
        const auto DH = q_shape[3];
        const auto Sk = k_shape[2];
        if (this->is_causal) {
            TT_FATAL(
                Sq == Sk, "Causal SDPA requires Q and K to have the same sequence length. Got Q: {}, K: {}", Sq, Sk);
        }

        if (use_mla) {
            // Head dim v validation
            TT_FATAL(
                this->head_dim_v <= q_shape[3],
                "Head dimension of V must be less than or equal to head dim of Q, got {} and {}",
                head_dim_v,
                q_shape[3]);
            TT_FATAL(
                this->head_dim_v <= k_shape[3],
                "Head dimension of V must be less than or equal to head dim of K, got {} and {}",
                this->head_dim_v,
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

        if (this->program_config.has_value()) {
            auto q_chunk_size = program_config->q_chunk_size;
            auto k_chunk_size = program_config->k_chunk_size;

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
        TT_FATAL(chunk_start_idx.has_value(), "chunk_start_idx must be provided for chunked mode");
        TT_FATAL(chunk_start_idx.value() >= 0, "chunk_start_idx must be non-negative");

        // Validate page table tensor
        const auto& page_table = optional_input_tensors[1].value();
        TT_FATAL(page_table.storage_type() == StorageType::DEVICE, "Page table tensor must be on device");
        TT_FATAL(
            input_tensors.at(0).device() == page_table.device(),
            "Page table must be on the same device as the input tensors");
        TT_FATAL(page_table.layout() == Layout::ROW_MAJOR, "Page table must be row major");
        // Check that page table is int32
        TT_FATAL(page_table.dtype() == DataType::INT32, "Page table must be int32");
        // Validate that first optional tensor (mask) is not provided
        TT_FATAL(
            !optional_input_tensors[0].has_value(),
            "Attention mask should not be provided in chunked mode - masking is handled internally");

        // Additional chunked-specific validations
        const auto q_shape = input_tensors.at(0).logical_shape();
        const auto k_shape = input_tensors.at(1).logical_shape();
        const auto v_shape = use_mla ? input_tensors.at(1).logical_shape() : input_tensors.at(2).logical_shape();
        const auto page_table_shape = page_table.logical_shape();
        const auto B = q_shape[0];
        const auto nqh = q_shape[1];
        const auto nkv = k_shape[1];
        const auto Sq = q_shape[2];
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

        if (this->program_config.has_value()) {
            auto q_chunk_size = program_config->q_chunk_size;
            auto k_chunk_size = program_config->k_chunk_size;

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

        // In chunked mode, K's sequence dimension should be >= Q's sequence dimension + chunk_start_idx
        TT_FATAL(
            kv_length >= q_shape[2] + chunk_start_idx.value(),
            "K's sequence length must be >= Q's sequence length + chunk_start_idx. Got K: {}, Q: {}, chunk_start_idx: "
            "{}",
            kv_length,
            q_shape[2],
            chunk_start_idx.value());
    };

    auto check_conditions = [&]() {
        bool has_chunk_start = chunk_start_idx.has_value();
        bool has_two_optional_inputs = optional_input_tensors.size() == 2;
        bool has_page_table = optional_input_tensors.size() > 1 && optional_input_tensors.at(1).has_value();
        TT_FATAL(
            has_chunk_start == has_two_optional_inputs, "chunk_start_idx and number of optional inputs must match");
        TT_FATAL(
            has_two_optional_inputs == has_page_table,
            "page_table must be provided if and only if there are two optional inputs");
    };

    check_conditions();
    bool is_chunked_mode = chunk_start_idx.has_value();

    // Check if we're in chunked mode and call appropriate validation
    if (is_chunked_mode) {
        validate_chunked_mode();
    } else {
        validate_regular_mode();
    }

    // Check padding: Only the sequence dimension may be padded. For all other dims, logical shape must be equal to
    // legacy shape
    for (const auto& tensor : input_tensors) {
        validate_padding(tensor);
    }
}

std::vector<TensorSpec> ScaledDotProductAttention::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    auto shape = input.logical_shape();
    if (use_mla) {
        shape[3] = this->head_dim_v.value_or(shape[3]);
    }
    return {TensorSpec(shape, TensorLayout(input.dtype(), PageConfig(Layout::TILE), output_mem_config))};
}

std::uint32_t ScaledDotProductAttention::get_q_chunk_size() const {
    return this->program_config ? this->program_config->q_chunk_size : 32;
}

std::uint32_t ScaledDotProductAttention::get_k_chunk_size() const {
    return this->program_config ? this->program_config->k_chunk_size : 32;
}

operation::ProgramWithCallbacks ScaledDotProductAttention::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = this->use_mla.value_or(false) ? input_tensors.at(1) : input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    const auto& attn_mask = optional_input_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1]));
    }

    std::size_t q_chunk_size = this->get_q_chunk_size();
    std::size_t k_chunk_size = this->get_k_chunk_size();

    // get page table if chunked
    const auto page_table = this->chunk_start_idx.has_value() ? optional_input_tensors.at(1) : std::nullopt;

    return detail::sdpa_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        output_tensor,
        attn_mask,
        page_table,
        this->chunk_start_idx,
        scale,
        this->is_causal,
        q_chunk_size,
        k_chunk_size,
        this->compute_kernel_config,
        this->program_config,
        this->use_mla.value_or(false),
        this->head_dim_v.value_or(0));
}

operation::OpPerformanceModel ScaledDotProductAttention::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = this->use_mla.value_or(false) ? input_tensors.at(1) : input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    if (output_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    // calculate arch specific parameters
    MathFidelity math_fidelity = ttnn::get_math_fidelity(compute_kernel_config);
    auto arch = output_tensor.storage_type() == StorageType::DEVICE
                    ? output_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "SDPA perf model does not support tt::arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModel(input_tensors, output_tensors, 0);
    }

    // Get main dimensions for Q*K and softmax(QK^T/sqrt) * V matmuls
    auto q_shape = input_tensor_q.logical_shape();
    auto k_shape = input_tensor_k.logical_shape();
    auto v_shape = input_tensor_v.logical_shape();
    TT_ASSERT(q_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor Q rank != 4");
    TT_ASSERT(k_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor K rank != 4");
    if (!use_mla) {
        TT_ASSERT(v_shape.size() == 4, "ScaledDotProductAttention perf model: input tensor V rank != 4");
    }

    bool is_chunked_prefill = this->chunk_start_idx.has_value();

    uint32_t batch_size_q = q_shape[0];
    uint32_t batch_size_k = k_shape[0];
    // NB: number of Q heads determines the shape of first matmul; K is broadcast to match Q's shape
    uint32_t num_heads_q = q_shape[1];

    const auto Sq = q_shape[2];
    const auto Sk = (is_chunked_prefill) ? q_shape[-2] + chunk_start_idx.value() : k_shape[2];
    const auto DH = q_shape[3];

    uint32_t Sv, DV;
    if (use_mla) {
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
    if (this->is_causal) {
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

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

operation::Hash ScaledDotProductAttention::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    bool is_chunked_prefill = this->chunk_start_idx.has_value();
    return operation::hash_operation<ScaledDotProductAttention>(
        this->head_dim_v,
        this->scale,
        this->output_mem_config,
        this->program_config,
        this->is_causal,
        is_chunked_prefill,
        this->compute_kernel_config,
        input_tensors,
        optional_input_tensors);
}

}  // namespace ttnn::operations::transformer
