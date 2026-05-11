// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_nanobind.hpp"

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/fft/fft.hpp"

namespace ttnn::operations::experimental::fft_binding::detail {

namespace {

using PrimPrecision = ttnn::experimental::prim::FFTPrecision;

// String → FFTPrecision (case-sensitive). Anything else → ValueError on the Python side.
PrimPrecision parse_precision(const std::string& s) {
    if (s == "precise") {
        return PrimPrecision::Precise;
    }
    if (s == "fast") {
        return PrimPrecision::Fast;
    }
    throw std::invalid_argument(
        "ttnn.experimental.fft / ttnn.experimental.ifft: precision must be 'precise' or 'fast' "
        "(got '" + s + "').");
}

// ---- Trampolines (stateless lambdas decay to plain function pointers,
//      which is what ttnn::bind_function / overload_t expects) ----

using TensorPair = std::tuple<ttnn::Tensor, ttnn::Tensor>;

TensorPair fft_real_trampoline(const ttnn::Tensor& input_real, std::string precision) {
    return ttnn::operations::experimental::fft(input_real, parse_precision(precision));
}

TensorPair fft_complex_trampoline(
    const ttnn::Tensor& input_real, const ttnn::Tensor& input_imag, std::string precision) {
    return ttnn::operations::experimental::fft(input_real, input_imag, parse_precision(precision));
}

TensorPair ifft_trampoline(
    const ttnn::Tensor& spectrum_real, const ttnn::Tensor& spectrum_imag, std::string precision) {
    return ttnn::operations::experimental::ifft(spectrum_real, spectrum_imag, parse_precision(precision));
}

}  // namespace

void bind_experimental_fft_operation(nb::module_& mod) {
    const auto* fft_doc =
        R"doc(
            1-D Fast Fourier Transform (forward).

            Computes the discrete Fourier transform of the last dimension of
            ``input_real``. Leading dimensions are batched. Returns a pair
            ``(real, imag)`` of tensors with the same shape as the input,
            holding the natural-order complex spectrum
            ``X[0], X[1], ..., X[N-1]``.

            Equivalent PyTorch:

            .. code-block:: python

                X = torch.fft.fft(input_real)   # complex64
                real, imag = X.real, X.imag

            Args:
                * :attr:`input_real`: Float32 or BFloat16 ROW_MAJOR tensor.
                * :attr:`input_imag` (optional): same shape, dtype, layout as
                  ``input_real``. When supplied, the input is treated as the
                  complex signal ``input_real + i * input_imag``; when omitted,
                  the imaginary part is taken to be zero.
                * :attr:`precision` (str, default ``"precise"``):
                  ``"precise"`` → SFPU true-fp32 path (matches ``torch.fft``
                  precision; round-trip ~1e-7).
                  ``"fast"`` → FPU bf16-mantissa matmul (faster for small N
                  but ~1e-3 round-trip). Only meaningful for Float32 + non-pow2
                  N; ignored everywhere else.

            Returns:
                Tuple ``(real, imag)`` of Tensors.

            Examples::

                # Real input (most common; precise default matches torch):
                spec_re, spec_im = ttnn.experimental.fft(x_real)

                # Opt into the fast (bf16-mantissa) path for small N:
                spec_re, spec_im = ttnn.experimental.fft(x_real, precision="fast")

                # Complex input — chaining FFT after another op that already
                # produced a (real, imag) pair:
                spec_re, spec_im = ttnn.experimental.fft(x_real, x_imag)
        )doc";

    ttnn::bind_function<"fft", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        fft_doc,
        ttnn::overload_t(
            &fft_real_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("precision") = std::string("precise")),
        ttnn::overload_t(
            &fft_complex_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("input_imag").noconvert(),
            nb::arg("precision") = std::string("precise")));

    const auto* ifft_doc =
        R"doc(
            1-D Inverse Fast Fourier Transform.

            Reverses :func:`ttnn.experimental.fft`. Takes the (real, imag)
            halves of a spectrum and returns the (real, imag) of the
            reconstructed signal scaled by 1/N. For a real input ``x``::

                spec_re, spec_im = ttnn.experimental.fft(x)
                rec_re,  rec_im  = ttnn.experimental.ifft(spec_re, spec_im)
                # rec_re == x  (within fp32 noise);  rec_im ~ 0

            Args:
                * :attr:`spectrum_real`: Float32 or BFloat16 ROW_MAJOR tensor, real part.
                * :attr:`spectrum_imag`: Same shape/dtype/layout as ``spectrum_real``.
                * :attr:`precision` (str, default ``"precise"``): same selector
                  as :func:`ttnn.experimental.fft`.

            Returns:
                Tuple ``(real, imag)`` of Tensors, same shape as the inputs.
        )doc";

    ttnn::bind_function<"ifft", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        ifft_doc,
        ttnn::overload_t(
            &ifft_trampoline,
            nb::arg("spectrum_real").noconvert(),
            nb::arg("spectrum_imag").noconvert(),
            nb::arg("precision") = std::string("precise")));
}

}  // namespace ttnn::operations::experimental::fft_binding::detail
