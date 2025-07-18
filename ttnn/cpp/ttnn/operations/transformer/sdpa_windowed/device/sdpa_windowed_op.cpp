// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_windowed_op.hpp"
#include <sys/types.h>

#include "sdpa_windowed_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

void WindowedScaledDotProductAttention::validate(const std::vector<Tensor>& input_tensors) const {
    // Common validations for windowed SDPA
    TT_FATAL(input_tensors.size() == 4, "Must have 4 input tensors (Q, K, V, cu_window_seqlens)");

    // Check cu_window_seqlens shape
    const auto& cu_window_seqlens = input_tensors.at(3);
    const auto& cu_window_seqlens_shape = cu_window_seqlens.logical_shape();
    TT_FATAL(cu_window_seqlens_shape.rank() == 1, "cu_window_seqlens must be a 1D tensor");
    TT_FATAL(cu_window_seqlens_shape[0] >= 2, "cu_window_seqlens must have at least 2 elements");
    TT_FATAL(
        cu_window_seqlens_shape[0] <= cu_window_seqlens_npages * cu_window_seqlens_page_size,
        "cu_window_seqlens must have less than 1024 elements");
    // First element must be 0
    // todo)) use uint16_t and int16_t instead of uint32_t and int32_t
    TT_FATAL(
        cu_window_seqlens.dtype() == DataType::UINT32 || cu_window_seqlens.dtype() == DataType::INT32,
        "cu_window_seqlens must be uint32 or int32");
    if (cu_window_seqlens.dtype() == DataType::UINT32) {
        const auto cu_window_seqlens_host = cu_window_seqlens.cpu().to_vector<uint32_t>();
        TT_FATAL(cu_window_seqlens_host.at(0) == 0, "First element of cu_window_seqlens must be 0");
    } else {
        const auto cu_window_seqlens_host = cu_window_seqlens.cpu().to_vector<int32_t>();
        TT_FATAL(cu_window_seqlens_host.at(0) == 0, "First element of cu_window_seqlens must be 0");
    }

    // Check storage and dtype
    for (size_t i = 0; i < 4; ++i) {
        const auto& input_tensor = input_tensors[i];
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to windowed SDPA need to be on device");
        TT_FATAL(
            input_tensor.buffer() != nullptr, "Operands to windowed SDPA need to be allocated in buffers on device");
        TT_FATAL(
            input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to windowed SDPA need to be in DRAM");
        if (i < 3) {
            TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to windowed SDPA must be tilized");
            TT_FATAL(
                input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B ||
                    input_tensor.dtype() == DataType::BFLOAT4_B,
                "Data type of input tensor must be BFLOAT16, BFLOAT8_B, or BFLOAT4_B and is {}",
                input_tensor.dtype());
        } else {
            TT_FATAL((input_tensor.layout() == Layout::ROW_MAJOR), "cu_window_seqlens must be in row-major layout");
        }
    }

    auto validate_padding = [&](const Tensor& tensor) {
        auto logical_shape = tensor.logical_shape();
        auto padded_shape = tensor.padded_shape();
        TT_FATAL(
            logical_shape[0] == padded_shape[0],
            "Padding is not supported on the batch dimension and is {}",
            logical_shape[0]);
        TT_FATAL(
            logical_shape[1] == padded_shape[1],
            "Padding is not supported on the num_heads dimension and is {}",
            logical_shape[1]);
        TT_FATAL(
            logical_shape[3] == padded_shape[3],
            "Padding is not supported on the head_dim dimension and is {}",
            logical_shape[3]);
    };

    auto validate_regular_mode = [&]() {
        // Shape checks
        const auto q_shape = input_tensors.at(0).logical_shape();
        const auto k_shape = input_tensors.at(1).logical_shape();
        const auto v_shape = input_tensors.at(2).logical_shape();
        const auto B = q_shape[0];
        const auto nqh = q_shape[1];
        const auto nkv = k_shape[1];
        const auto Sq = q_shape[2];
        const auto DH = q_shape[3];
        const auto Sk = k_shape[2];

        TT_FATAL(
            k_shape[0] == B && v_shape[0] == B, "K and V batch must match. Got K: {}, V: {}", k_shape[0], v_shape[0]);
        TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
        TT_FATAL(v_shape[2] == Sk, "K and V sequence length must match. Got K: {}, V: {}", k_shape[2], v_shape[2]);
        TT_FATAL(
            k_shape[3] == DH && v_shape[3] == DH,
            "K and V hidden dim must match. Got K: {}, V: {}",
            k_shape[3],
            v_shape[3]);
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

    validate_regular_mode();

    // Check padding: Only the sequence dimension may be padded. For all other dims, logical shape must be equal to
    // padded shape
    validate_padding(input_tensors.at(0));
    validate_padding(input_tensors.at(1));
    validate_padding(input_tensors.at(2));
}

std::vector<TensorSpec> WindowedScaledDotProductAttention::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    return {
        TensorSpec(input.logical_shape(), TensorLayout(input.dtype(), PageConfig(Layout::TILE), output_mem_config))};
}

std::uint32_t WindowedScaledDotProductAttention::get_q_chunk_size() const {
    return this->program_config ? this->program_config->q_chunk_size : 32;
}

std::uint32_t WindowedScaledDotProductAttention::get_k_chunk_size() const {
    return this->program_config ? this->program_config->k_chunk_size : 32;
}

operation::ProgramWithCallbacks WindowedScaledDotProductAttention::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& cu_window_seqlens = input_tensors.at(3);
    auto& output_tensor = output_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1]));
    }

    std::size_t q_chunk_size = this->get_q_chunk_size();
    std::size_t k_chunk_size = this->get_k_chunk_size();

    return detail::sdpa_windowed_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cu_window_seqlens,
        output_tensor,
        scale,
        q_chunk_size,
        k_chunk_size,
        this->compute_kernel_config,
        this->program_config);
}

operation::OpPerformanceModel WindowedScaledDotProductAttention::create_op_performance_model(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    // Similar to regular SDPA performance model but accounting for windowed pattern
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& cu_window_seqlens = input_tensors.at(3);
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
        log_warning(tt::LogOp, "Windowed SDPA perf model does not support tt::arch '{}'", magic_enum::enum_name(arch));
        return operation::OpPerformanceModel(input_tensors, output_tensors, 0);
    }

    // Get main dimensions
    auto q_shape = input_tensor_q.logical_shape();
    auto k_shape = input_tensor_k.logical_shape();
    auto v_shape = input_tensor_v.logical_shape();

    uint32_t batch_size = q_shape[0];
    uint32_t num_heads_q = q_shape[1];
    const auto Sq = q_shape[2];
    const auto Sk = k_shape[2];
    const auto Sv = v_shape[3];
    const auto DH = q_shape[3];
    const auto DV = v_shape[2];

    // For windowed attention, we only compute attention within windows
    // Calculate total number of attention computations based on windows
    int64_t num_mul_adds = 0;
    constexpr int64_t FLOPS_PER_FMA = 2;

    CoreCoord compute_grid_dims = output_tensor.device()->compute_with_storage_grid_size();
    int num_cores = compute_grid_dims.x * compute_grid_dims.y;

    const int tensix_mul_adds_per_cycle_lofi = 4096;

    int ideal_dev_clock_cycles = std::ceil(
        ((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        (float)operation::OpPerformanceModel::fidelity_multiplier(math_fidelity));

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

operation::Hash WindowedScaledDotProductAttention::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    return operation::hash_operation<WindowedScaledDotProductAttention>(
        this->scale, this->output_mem_config, this->program_config, this->compute_kernel_config, input_tensors);
}

}  // namespace ttnn::operations::transformer
