// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include <cmath>
#include <vector>

namespace ttnn {

namespace {

Tensor create_dft_matrix(
    uint32_t N,
    bool inverse,
    bool real_part,
    DataType dtype,
    Layout layout,
    tt::tt_metal::distributed::MeshDevice* device,
    const MemoryConfig& mem_config) {
    uint32_t size = N * N;
    TensorSpec spec(
        ttnn::Shape(std::vector<uint32_t>{N, N}),
        TensorLayout(dtype, PageConfig(layout), mem_config)
    );
    constexpr double PI = 3.14159265358979323846;
    if (dtype == DataType::FLOAT32) {
        std::vector<float> owned_buffer(size);
        for (uint32_t k = 0; k < N; ++k) {
            for (uint32_t n = 0; n < N; ++n) {
                double angle = 2.0 * PI * k * n / N;
                if (inverse) {
                    if (real_part) {
                        owned_buffer[k * N + n] = static_cast<float>(std::cos(angle) / N);
                    } else {
                        owned_buffer[k * N + n] = static_cast<float>(std::sin(angle) / N);
                    }
                } else {
                    if (real_part) {
                        owned_buffer[k * N + n] = static_cast<float>(std::cos(angle));
                    } else {
                        owned_buffer[k * N + n] = static_cast<float>(-std::sin(angle));
                    }
                }
            }
        }
        return Tensor::from_vector(std::move(owned_buffer), spec, device);
    } else {
        std::vector<bfloat16> owned_buffer(size);
        for (uint32_t k = 0; k < N; ++k) {
            for (uint32_t n = 0; n < N; ++n) {
                double angle = 2.0 * PI * k * n / N;
                if (inverse) {
                    if (real_part) {
                        owned_buffer[k * N + n] = bfloat16(static_cast<float>(std::cos(angle) / N));
                    } else {
                        owned_buffer[k * N + n] = bfloat16(static_cast<float>(std::sin(angle) / N));
                    }
                } else {
                    if (real_part) {
                        owned_buffer[k * N + n] = bfloat16(static_cast<float>(std::cos(angle)));
                    } else {
                        owned_buffer[k * N + n] = bfloat16(static_cast<float>(-std::sin(angle)));
                    }
                }
            }
        }
        return Tensor::from_vector(std::move(owned_buffer), spec, device);
    }
}

ComplexTensor fft_impl(
    const ComplexTensor& input_tensor,
    int64_t dim,
    bool inverse,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    Tensor real = input_tensor.real();
    Tensor imag = input_tensor.imag();
    auto shape = real.logical_shape();
    int64_t rank = shape.rank();
    int64_t dim_normalized = dim < 0 ? dim + rank : dim;
    uint32_t N = shape[dim_normalized];
    auto* device = real.device();

    bool need_transpose = (dim_normalized != rank - 1);
    Tensor real_trans = need_transpose ? ttnn::transpose(real, dim_normalized, -1) : real;
    Tensor imag_trans = need_transpose ? ttnn::transpose(imag, dim_normalized, -1) : imag;

    auto original_layout = real_trans.layout();
    Tensor real_tile = (original_layout != Layout::TILE) ? ttnn::to_layout(real_trans, Layout::TILE, std::nullopt, std::nullopt, device) : real_trans;
    Tensor imag_tile = (original_layout != Layout::TILE) ? ttnn::to_layout(imag_trans, Layout::TILE, std::nullopt, std::nullopt, device) : imag_trans;

    auto dtype = real_tile.dtype();
    auto output_mem_config = memory_config.value_or(real_tile.memory_config());
    Tensor W_real = create_dft_matrix(N, inverse, true, dtype, Layout::TILE, device, output_mem_config);
    Tensor W_imag = create_dft_matrix(N, inverse, false, dtype, Layout::TILE, device, output_mem_config);

    Tensor A_W_real = ttnn::matmul(real_tile, W_real, std::nullopt, output_mem_config);
    Tensor B_W_imag = ttnn::matmul(imag_tile, W_imag, std::nullopt, output_mem_config);
    Tensor Y_real_tile = ttnn::subtract(A_W_real, B_W_imag, std::nullopt, output_mem_config);
    A_W_real.deallocate();
    B_W_imag.deallocate();

    Tensor A_W_imag = ttnn::matmul(real_tile, W_imag, std::nullopt, output_mem_config);
    Tensor B_W_real = ttnn::matmul(imag_tile, W_real, std::nullopt, output_mem_config);
    Tensor Y_imag_tile = ttnn::add(A_W_imag, B_W_real, std::nullopt, output_mem_config);
    A_W_imag.deallocate();
    B_W_real.deallocate();

    W_real.deallocate();
    W_imag.deallocate();

    if (original_layout != Layout::TILE) {
        if (real_trans.layout() != Layout::TILE) {
            real_tile.deallocate();
            imag_tile.deallocate();
        }
    }

    Tensor Y_real_trans = (original_layout != Layout::TILE) ? ttnn::to_layout(Y_real_tile, original_layout, std::nullopt, std::nullopt, device) : Y_real_tile;
    Tensor Y_imag_trans = (original_layout != Layout::TILE) ? ttnn::to_layout(Y_imag_tile, original_layout, std::nullopt, std::nullopt, device) : Y_imag_tile;

    if (original_layout != Layout::TILE) {
        Y_real_tile.deallocate();
        Y_imag_tile.deallocate();
    }

    Tensor Y_real = need_transpose ? ttnn::transpose(Y_real_trans, dim_normalized, -1) : Y_real_trans;
    Tensor Y_imag = need_transpose ? ttnn::transpose(Y_imag_trans, dim_normalized, -1) : Y_imag_trans;

    if (need_transpose) {
        Y_real_trans.deallocate();
        Y_imag_trans.deallocate();
        real_trans.deallocate();
        imag_trans.deallocate();
    }

    return complex_tensor(Y_real, Y_imag);
}

} // namespace

ComplexTensor fft(
    const ComplexTensor& input_tensor,
    int64_t dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    return fft_impl(input_tensor, dim, false, memory_config);
}

ComplexTensor fft(
    const Tensor& input_tensor,
    int64_t dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    Tensor imag = ttnn::zeros_like(input_tensor, std::nullopt, std::nullopt, std::nullopt, memory_config);
    ComplexTensor complex_input = complex_tensor(input_tensor, imag);
    ComplexTensor result = fft(complex_input, dim, memory_config);
    return result;
}

ComplexTensor ifft(
    const ComplexTensor& input_tensor,
    int64_t dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    return fft_impl(input_tensor, dim, true, memory_config);
}

ComplexTensor ifft(
    const Tensor& input_tensor,
    int64_t dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    Tensor imag = ttnn::zeros_like(input_tensor, std::nullopt, std::nullopt, std::nullopt, memory_config);
    ComplexTensor complex_input = complex_tensor(input_tensor, imag);
    ComplexTensor result = ifft(complex_input, dim, memory_config);
    return result;
}

} // namespace ttnn
