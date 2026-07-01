// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-op skeleton for ttnn::prim::apply_twiddles — the between-pass
// elementwise complex-multiply step of Cooley–Tukey two-pass FFT.

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "apply_twiddles_device_operation_types.hpp"
#include "apply_twiddles_factory.hpp"

namespace ttnn::experimental::prim {

struct ApplyTwiddlesDeviceOperation {
    using operation_attributes_t = ApplyTwiddlesParams;
    using tensor_args_t          = ApplyTwiddlesTensorArgs;

    // 2-tuple (real, imag), see fft_device_operation.hpp for the
    // tuple-vs-pair rationale.
    using spec_return_value_t   = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    using program_factory_t = std::variant<ApplyTwiddlesFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Public entry point used by the op-API layer (apply_twiddles.cpp).
std::tuple<Tensor, Tensor> apply_twiddles(
    const Tensor& input_real,
    const Tensor& input_imag,
    uint32_t N1,
    uint32_t N2);

}  // namespace ttnn::prim
