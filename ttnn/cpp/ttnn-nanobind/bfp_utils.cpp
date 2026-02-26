// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bfp_utils.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt_stl/span.hpp>

namespace ttnn::bfp_utils {

namespace nb = nanobind;

// Pack float32 data into BFP tiles, return raw uint32 packed data as numpy array.
template <typename PackFn>
static nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>> pack_impl(
    PackFn pack_fn, nb::ndarray<nb::numpy, const float, nb::ndim<1>> input, bool row_major_input, bool is_exp_a) {
    tt::stl::Span<const float> data_span(input.data(), input.size());
    auto packed = pack_fn(data_span, row_major_input, is_exp_a, std::nullopt);

    auto* result = new uint32_t[packed.size()];
    std::copy(packed.begin(), packed.end(), result);

    nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<uint32_t*>(p); });
    size_t shape[] = {packed.size()};
    return nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(result, 1, shape, std::move(owner));
}

// Unpack raw uint32 BFP tile data back to float32 numpy array.
template <typename UnpackFn>
static nb::ndarray<nb::numpy, float, nb::ndim<1>> unpack_impl(
    UnpackFn unpack_fn,
    nb::ndarray<nb::numpy, const uint32_t, nb::ndim<1>> input,
    bool row_major_output,
    bool is_exp_a) {
    tt::stl::Span<const uint32_t> data_span(input.data(), input.size());
    auto unpacked = unpack_fn(data_span, row_major_output, is_exp_a, std::nullopt);

    auto* result = new float[unpacked.size()];
    std::copy(unpacked.begin(), unpacked.end(), result);

    nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    size_t shape[] = {unpacked.size()};
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(result, 1, shape, std::move(owner));
}

void py_module(nb::module_& mod) {
    mod.def(
        "pack_bfp8",
        [](nb::ndarray<nb::numpy, const float, nb::ndim<1>> input, bool row_major_input, bool is_exp_a) {
            return pack_impl(pack_as_bfp8_tiles<float>, input, row_major_input, is_exp_a);
        },
        nb::arg("input"),
        nb::arg("row_major_input") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Pack float32 data into BFP8 tile format. Returns raw uint32 packed data.)doc");

    mod.def(
        "pack_bfp4",
        [](nb::ndarray<nb::numpy, const float, nb::ndim<1>> input, bool row_major_input, bool is_exp_a) {
            return pack_impl(pack_as_bfp4_tiles<float>, input, row_major_input, is_exp_a);
        },
        nb::arg("input"),
        nb::arg("row_major_input") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Pack float32 data into BFP4 tile format. Returns raw uint32 packed data.)doc");

    mod.def(
        "unpack_bfp8",
        [](nb::ndarray<nb::numpy, const uint32_t, nb::ndim<1>> input, bool row_major_output, bool is_exp_a) {
            return unpack_impl(unpack_bfp8_tiles_into_float_vec, input, row_major_output, is_exp_a);
        },
        nb::arg("input"),
        nb::arg("row_major_output") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Unpack raw BFP8 tile data back to float32.)doc");

    mod.def(
        "unpack_bfp4",
        [](nb::ndarray<nb::numpy, const uint32_t, nb::ndim<1>> input, bool row_major_output, bool is_exp_a) {
            return unpack_impl(unpack_bfp4_tiles_into_float_vec, input, row_major_output, is_exp_a);
        },
        nb::arg("input"),
        nb::arg("row_major_output") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Unpack raw BFP4 tile data back to float32.)doc");
}

}  // namespace ttnn::bfp_utils
