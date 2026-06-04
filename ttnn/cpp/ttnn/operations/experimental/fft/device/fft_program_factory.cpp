// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FFT program factory — full-backend dispatcher.

#include "fft_program_factory.hpp"

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>

#include "stockham_host.hpp"
#include "universal_host.hpp"
#include "universal_xl_host.hpp"
#include "universal_bf16_host.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace ttnn::experimental::prim {

namespace {

using Complex  = std::complex<float>;
using DataType = tt::tt_metal::DataType;
using tt::tt_metal::distributed::MeshDevice;

// Renamed `is_pow2_local` (rather than `is_pow2`) so the Unity build can
// merge this TU with fft_device_operation.cpp — which has its own
// anonymous-namespace `is_pow2` — without ODR collision.
constexpr bool is_pow2_local(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

std::vector<Complex> fft_one_row(
    std::shared_ptr<MeshDevice>& md,
    DataType                     dtype,
    FFTPrecision                 precision,
    const std::vector<Complex>&  signal) {

    if (dtype == DataType::BFLOAT16) {
        return fft_universal_bf16::fft(md, signal);
    }
    // Float32
    const uint32_t N = static_cast<uint32_t>(signal.size());
    if (!is_pow2_local(N)) {
        return (precision == FFTPrecision::Fast)
            ? fft_universal::fft        (md, signal)
            : fft_universal::fft_precise(md, signal);
    }
    if (N <= 1u * 1024u * 1024u)     return fft_stockham::fft(md, signal);
    return fft_universal_xl::fft(md, signal);
}

// ── Tensor I/O helpers (handle fp32 ↔ bf16 at the host boundary) ────────────

std::vector<float> read_real_as_fp32(const Tensor& t) {
    if (t.dtype() == DataType::BFLOAT16) {
        const auto buf = t.to_vector<bfloat16>();
        std::vector<float> out(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) {
            out[i] = static_cast<float>(buf[i]);
        }
        return out;
    }
    return t.to_vector<float>();
}

// Build an output Tensor matching `spec` (dtype/shape/layout/memory) from
// a host fp32 buffer, narrowing to bf16 if the spec requires it. The
// returned tensor lives on `device`.
Tensor write_real_with_spec(
    std::vector<float>&&     buf,
    const ttnn::TensorSpec&  spec,
    MeshDevice*              device) {

    if (spec.data_type() == DataType::BFLOAT16) {
        std::vector<bfloat16> bf(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) {
            bf[i] = bfloat16(buf[i]);
        }
        return Tensor::from_vector(std::move(bf), spec, device);
    }
    return Tensor::from_vector(std::move(buf), spec, device);
}

void run_backend_fft(
    const FFTParams&            attrs,
    const FFTTensorArgs&        tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {

    const auto& in_re_tensor = tensor_args.input_real;
    const auto& shape        = in_re_tensor.logical_shape();
    TT_FATAL(shape.size() >= 1u, "fft: tensor has rank 0");

    const uint32_t N     = shape[-1];
    const uint64_t total = in_re_tensor.logical_volume();
    TT_FATAL(total % N == 0u,
             "fft: total volume {} not divisible by N {}.", total, N);
    const uint64_t batches = total / N;

    const auto dtype = in_re_tensor.dtype();

    // 1. Materialise host-side fp32 inputs. input_imag is OPTIONAL on the
    //    forward path (zero-fill when absent → real-input FFT) and REQUIRED
    //    on the inverse path (you can't IFFT a real-only spectrum).
    const std::vector<float> in_re = read_real_as_fp32(in_re_tensor);
    std::vector<float>       in_im;
    if (tensor_args.input_imag.has_value()) {
        in_im = read_real_as_fp32(*tensor_args.input_imag);
        TT_FATAL(in_im.size() == in_re.size(),
                 "fft: real / imag size mismatch ({} vs {}).",
                 in_re.size(), in_im.size());
    } else {
        TT_FATAL(!attrs.inverse,
                 "fft (inverse): input_imag is required.");
        in_im.assign(in_re.size(), 0.0f);
    }
    TT_FATAL(in_re.size() == total,
             "fft: read returned {} elements, expected {}.",
             in_re.size(), total);

    // 2. Wrap the device pointer in a no-op-deleter shared_ptr — the
    //    orchestrators all take `shared_ptr<MeshDevice>`, but we don't
    //    own the lifetime here (the tensor does).
    auto* device_raw = in_re_tensor.device();
    auto md = std::shared_ptr<MeshDevice>(
        device_raw, [](MeshDevice*){});

    std::vector<float>   out_re(total);
    std::vector<float>   out_im(total);
    std::vector<Complex> work(N);

    // IFFT via conjugate trick: y = conj(fft(conj(X))) / N.
    const float scale = attrs.inverse ? (1.0f / static_cast<float>(N)) : 1.0f;

    // 3. Per-row dispatch through the selected backend.
    for (uint64_t b = 0u; b < batches; ++b) {
        const uint64_t off = b * static_cast<uint64_t>(N);

        if (attrs.inverse) {
            for (uint32_t i = 0u; i < N; ++i) {
                work[i] = Complex{in_re[off + i], -in_im[off + i]};
            }
        } else {
            for (uint32_t i = 0u; i < N; ++i) {
                work[i] = Complex{in_re[off + i], in_im[off + i]};
            }
        }

        const auto X = fft_one_row(md, dtype, attrs.precision, work);

        if (attrs.inverse) {
            for (uint32_t k = 0u; k < N; ++k) {
                out_re[off + k] =  X[k].real() * scale;
                out_im[off + k] = -X[k].imag() * scale;
            }
        } else {
            for (uint32_t k = 0u; k < N; ++k) {
                out_re[off + k] = X[k].real();
                out_im[off + k] = X[k].imag();
            }
        }
    }

    // 4. Replace outputs with fresh device tensors. compute_output_specs
    //    guarantees output spec mirrors input spec (dtype/shape/layout).
    const auto& spec = in_re_tensor.tensor_spec();
    std::get<0>(tensor_return_value) = write_real_with_spec(std::move(out_re), spec, device_raw);
    std::get<1>(tensor_return_value) = write_real_with_spec(std::move(out_im), spec, device_raw);
}

}  // namespace

FFTProgramFactory::cached_program_t FFTProgramFactory::create(
    const FFTParams&            operation_attributes,
    const FFTTensorArgs&        tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {

    // Run the dispatched backend. The orchestrator handles its own
    // EnqueueMeshWorkload + read-back, so by the time this returns the
    // spectrum is already populated on `tensor_return_value`.
    run_backend_fft(operation_attributes, tensor_args, tensor_return_value);

    // Empty outer Program — the FFT work happens inside the orchestrator
    // calls above, not in a Program owned by this factory.
    tt::tt_metal::Program program{};
    const uint32_t N = tensor_args.input_real.logical_shape()[-1];

    return cached_program_t{
        std::move(program),
        FFTSharedVariables{
            .kernel_ids = {},
            .cores      = {},
            .N          = N,
        }};
}

void FFTProgramFactory::override_runtime_arguments(
    cached_program_t&           /*cached_program*/,
    const FFTParams&            operation_attributes,
    const FFTTensorArgs&        tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {
    // Cache-hit path — re-dispatch the backend. The cached "program" is
    // empty; there are no kernel runtime args to update.
    run_backend_fft(operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim

