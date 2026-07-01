// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-op skeleton for ttnn::prim::apply_twiddles_xl — large-modulus
// between-pass elementwise complex multiply used by fft_three_pass.

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "apply_twiddles_xl_device_operation_types.hpp"
#include "apply_twiddles_xl_factory.hpp"

namespace ttnn::experimental::prim {

struct ApplyTwiddlesXlDeviceOperation {
    using operation_attributes_t = ApplyTwiddlesXlParams;
    using tensor_args_t          = ApplyTwiddlesXlTensorArgs;

    using spec_return_value_t   = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    using program_factory_t = std::variant<ApplyTwiddlesXlFactory>;

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

// Public entry point used by the op-API layer (apply_twiddles_xl.cpp).
//   P            : row length (= last dim of input, pow-2 in [2, 1024])
//   big_modulus  : twiddle row modulus.  Pow-2, any size >= 1.  The three-
//                  pass composite uses N1*N2 which spans [2^10, 2^20] for
//                  cube-balanced N up to 2^30.  M must be a multiple of
//                  big_modulus.
//   full_N       : denominator of the angle.  Pow-2, >= big_modulus.  The
//                  three-pass composite uses full_N = N = P * big_modulus.
std::tuple<Tensor, Tensor> apply_twiddles_xl(
    const Tensor& input_real,
    const Tensor& input_imag,
    uint32_t P,
    uint32_t big_modulus,
    uint32_t full_N);

}  // namespace ttnn::prim
