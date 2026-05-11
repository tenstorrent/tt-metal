// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "fft_device_operation_types.hpp"
#include "fft_program_factory.hpp"

namespace ttnn::experimental::prim {

struct FFTDeviceOperation {
    using operation_attributes_t = FFTParams;
    using tensor_args_t          = FFTTensorArgs;

    // We return TWO tensors (real + imag of the FFT/IFFT output).
    // tuple, not pair: tt_stl reflection has no specialization for std::pair.
    using spec_return_value_t   = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    using program_factory_t = std::variant<FFTProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Public entry point used by the op-API layer (fft.cpp).
std::tuple<Tensor, Tensor> fft(
    const Tensor& input_real,
    bool inverse,
    const std::optional<Tensor>& input_imag,
    ttnn::experimental::prim::FFTPrecision precision =
        ttnn::experimental::prim::FFTPrecision::Precise);

}  // namespace ttnn::prim
