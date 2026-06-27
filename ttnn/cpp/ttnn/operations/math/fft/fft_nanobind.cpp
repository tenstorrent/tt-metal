// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_nanobind.hpp"
#include "fft.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

namespace ttnn::operations::math::fft {

namespace nb = nanobind;

void py_module(nb::module_& mod) {
    mod.def(
        "fft",
        nb::overload_cast<const ComplexTensor&, int64_t, const std::optional<tt::tt_metal::MemoryConfig>&>(&ttnn::fft),
        nb::arg("input_tensor"),
        nb::arg("dim") = -1,
        nb::arg("memory_config") = std::nullopt,
        "Compute 1D Fast Fourier Transform for complex tensor");

    mod.def(
        "fft",
        nb::overload_cast<const Tensor&, int64_t, const std::optional<tt::tt_metal::MemoryConfig>&>(&ttnn::fft),
        nb::arg("input_tensor"),
        nb::arg("dim") = -1,
        nb::arg("memory_config") = std::nullopt,
        "Compute 1D Fast Fourier Transform for real tensor");

    mod.def(
        "ifft",
        nb::overload_cast<const ComplexTensor&, int64_t, const std::optional<tt::tt_metal::MemoryConfig>&>(&ttnn::ifft),
        nb::arg("input_tensor"),
        nb::arg("dim") = -1,
        nb::arg("memory_config") = std::nullopt,
        "Compute 1D Inverse Fast Fourier Transform for complex tensor");

    mod.def(
        "ifft",
        nb::overload_cast<const Tensor&, int64_t, const std::optional<tt::tt_metal::MemoryConfig>&>(&ttnn::ifft),
        nb::arg("input_tensor"),
        nb::arg("dim") = -1,
        nb::arg("memory_config") = std::nullopt,
        "Compute 1D Inverse Fast Fourier Transform for real tensor");
}

}  // namespace ttnn::operations::math::fft
