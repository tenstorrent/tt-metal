// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-op skeleton for ttnn::prim::fft_radix_pass — fused batched
// length-P FFT + optional post-twiddle complex multiply.  Single
// dispatch, single ProgramDescriptor, trace-safe.

#pragma once

#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "fft_radix_pass_device_operation_types.hpp"
#include "fft_radix_pass_factory.hpp"

namespace ttnn::experimental::prim {

struct FftRadixPassDeviceOperation {
    using operation_attributes_t = FftRadixPassParams;
    using tensor_args_t          = FftRadixPassTensorArgs;

    // 2-tuple (real, imag) — same rationale as FFTDeviceOperation
    // (tt_stl reflection has no specialization for std::pair).
    using spec_return_value_t   = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    using program_factory_t = std::variant<FftRadixPassFactory>;

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

// Public entry point used by the op-API layer (fft_radix_pass.cpp).
//   P            : FFT length per row (= last dim of input, pow-2 in [2, 1024])
//   twiddle_N2   : 0 → pure FFT.  >0 → fused post-twiddle indexed by
//                       ((row / stride) % twiddle_N2).
//   stride       : Row-index stride for the post-twiddle lookup.  Default
//                  1 (= use row directly).  Used by fft_three_pass to skip
//                  an extra transpose between pass-2 and pass-3.
//   output_scale : Fused scalar multiplier applied to every output element
//                  (after the optional post-twiddle).  Default 1.0f =
//                  no-op (same kernel binary as commit 5).  Used by the
//                  IFFT path (commit 6c) to fold the 1/N scale into the
//                  LAST radix_pass writer of the composite, zero extra
//                  dispatch.
std::tuple<Tensor, Tensor> fft_radix_pass(
    const Tensor& input_real,
    const std::optional<Tensor>& input_imag,
    uint32_t P,
    uint32_t twiddle_N2,
    uint32_t stride       = 1,
    float    output_scale = 1.0f);

}  // namespace ttnn::prim
