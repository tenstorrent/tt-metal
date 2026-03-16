// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bfp_utils.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/bfloat2.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt_stl/span.hpp>

namespace ttnn::bfp_utils {

namespace nb = nanobind;

// Pack float32 data into BFP tiles, return raw uint32 packed data as numpy array.
// Input accepts any array_api-compatible source (numpy, torch, jax, etc.)
template <typename PackFn>
static nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>> pack_impl(
    const PackFn& pack_fn,
    const nb::ndarray<nb::array_api, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
    bool row_major_input,
    bool is_exp_a) {
    tt::stl::Span<const float> data_span(input.data(), input.size());
    auto packed = pack_fn(data_span, row_major_input, is_exp_a, std::nullopt);

    auto* result = new uint32_t[packed.size()];
    std::copy(packed.begin(), packed.end(), result);

    nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<uint32_t*>(p); });
    size_t shape[] = {packed.size()};
    return nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(result, 1, shape, std::move(owner));
}

// Unpack raw uint32 BFP tile data back to float32 numpy array.
// Input accepts any array_api-compatible source (numpy, torch, jax, etc.)
template <typename UnpackFn>
static nb::ndarray<nb::numpy, float, nb::ndim<1>> unpack_impl(
    const UnpackFn& unpack_fn,
    const nb::ndarray<nb::array_api, const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
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
        [](const nb::ndarray<nb::array_api, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_input,
           bool is_exp_a) { return pack_impl(pack_as_bfp8_tiles<float>, input, row_major_input, is_exp_a); },
        nb::arg("input"),
        nb::arg("row_major_input") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Pack float32 data into BFP8 tile format. Returns raw uint32 packed data.)doc");

    mod.def(
        "pack_bfp4",
        [](const nb::ndarray<nb::array_api, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_input,
           bool is_exp_a) { return pack_impl(pack_as_bfp4_tiles<float>, input, row_major_input, is_exp_a); },
        nb::arg("input"),
        nb::arg("row_major_input") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Pack float32 data into BFP4 tile format. Returns raw uint32 packed data.)doc");

    mod.def(
        "pack_bfp2",
        [](const nb::ndarray<nb::array_api, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_input,
           bool is_exp_a) {
            return pack_impl(tt::tt_metal::pack_as_bfp2_tiles<float>, input, row_major_input, is_exp_a);
        },
        nb::arg("input"),
        nb::arg("row_major_input") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Pack float32 data into BFP2 tile format. Returns raw uint32 packed data.)doc");

    mod.def(
        "unpack_bfp8",
        [](const nb::ndarray<nb::array_api, const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_output,
           bool is_exp_a) { return unpack_impl(unpack_bfp8_tiles_into_float_vec, input, row_major_output, is_exp_a); },
        nb::arg("input"),
        nb::arg("row_major_output") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Unpack raw BFP8 tile data back to float32.)doc");

    mod.def(
        "unpack_bfp4",
        [](const nb::ndarray<nb::array_api, const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_output,
           bool is_exp_a) { return unpack_impl(unpack_bfp4_tiles_into_float_vec, input, row_major_output, is_exp_a); },
        nb::arg("input"),
        nb::arg("row_major_output") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Unpack raw BFP4 tile data back to float32.)doc");

    mod.def(
        "unpack_bfp2",
        [](const nb::ndarray<nb::array_api, const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_output,
           bool is_exp_a) {
            return unpack_impl(tt::tt_metal::unpack_bfp2_tiles_into_float_vec, input, row_major_output, is_exp_a);
        },
        nb::arg("input"),
        nb::arg("row_major_output") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Unpack raw BFP2 tile data back to float32.)doc");

    mod.def("get_l1_alignment", &tt::tt_metal::hal::get_l1_alignment, R"doc(Get L1 memory alignment in bytes.)doc");
    mod.def(
        "get_dram_alignment", &tt::tt_metal::hal::get_dram_alignment, R"doc(Get DRAM memory alignment in bytes.)doc");

    mod.def(
        "compute_shard_page_mapping",
        [](std::vector<uint32_t> tensor_shape_vec,
           std::vector<uint32_t> shard_shape_vec,
           uint32_t page_height,
           uint32_t page_width,
           const tt::tt_metal::CoreRangeSet& core_range_set,
           tt::tt_metal::ShardOrientation shard_orientation,
           tt::tt_metal::ShardDistributionStrategy distribution_strategy)
            -> std::vector<std::pair<tt::tt_metal::CoreCoord, std::vector<uint32_t>>> {
            auto spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
                tt::tt_metal::Shape(tensor_shape_vec),
                tt::tt_metal::Shape(shard_shape_vec),
                tt::tt_metal::Shape2D(page_height, page_width),
                core_range_set,
                shard_orientation,
                distribution_strategy);
            auto mapping = spec.compute_page_mapping();
            std::vector<std::pair<tt::tt_metal::CoreCoord, std::vector<uint32_t>>> result;
            result.reserve(mapping.all_cores.size());
            for (size_t i = 0; i < mapping.all_cores.size(); i++) {
                std::vector<uint32_t> pages;
                for (auto p : mapping.core_host_page_indices[i]) {
                    if (p != tt::tt_metal::UncompressedBufferPageMapping::PADDING) {
                        pages.push_back(p);
                    }
                }
                result.emplace_back(mapping.all_cores[i], std::move(pages));
            }
            return result;
        },
        nb::arg("tensor_shape"),
        nb::arg("shard_shape"),
        nb::arg("page_height"),
        nb::arg("page_width"),
        nb::arg("core_range_set"),
        nb::arg("shard_orientation"),
        nb::arg("distribution_strategy") = tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D,
        R"doc(Compute shard-to-core page mapping using BufferDistributionSpec.

Returns list of (CoreCoord, page_indices) tuples, one per core.
Each page_index is a host page ID (for TILE_LAYOUT, page = tile).
Use ROUND_ROBIN_1D for HEIGHT/WIDTH sharding, GRID_2D for BLOCK sharding.
)doc");
}

}  // namespace ttnn::bfp_utils
