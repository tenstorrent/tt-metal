// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_supported.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::pad_codegen {

// Output CB depth is two batches, so reader and writer can each own one.
uint32_t cb_depth(uint32_t read_batch, uint32_t write_batch) { return 2 * std::max(read_batch, write_batch); }

namespace {

using tt::tt_metal::DataType;

constexpr uint32_t kReadBatch = 8;
constexpr uint32_t kWriteBatch = 4;

uint32_t bf16_bits(float v) {
    if (std::isnan(v)) {
        return 0x7FC0;
    }
    uint32_t b = std::bit_cast<uint32_t>(v);
    b += 0x7FFF + ((b >> 16) & 1);
    return b >> 16;
}

bool same_tile(const tt::tt_metal::Tile& a, const tt::tt_metal::Tile& b) {
    return a.get_height() == b.get_height() && a.get_width() == b.get_width() &&
           a.get_transpose_within_face() == b.get_transpose_within_face() &&
           a.get_transpose_of_faces() == b.get_transpose_of_faces();
}

// The tile-page math in the kernels assumes the default 32x32 untransposed tile.
bool tile_geometry_ok(const Tensor& input) {
    if (input.layout() != ttnn::TILE_LAYOUT) {
        return true;
    }
    const auto tile = input.tensor_spec().tile();
    return tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
           !tile.get_transpose_within_face() && !tile.get_transpose_of_faces();
}

bool dtype_ok(const Tensor& input) {
    switch (input.dtype()) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32: return true;
        case DataType::BFLOAT8_B: return input.layout() == ttnn::TILE_LAYOUT;
        default: return false;
    }
}

bool output_matches(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs, const Tensor& output) {
    if (attrs.fill_partial_tile) {
        return output.buffer() == input.buffer();
    }
    const auto expected = codegen_output_spec(input, attrs);
    return output.layout() == expected.layout() && output.dtype() == expected.data_type() &&
           same_tile(output.tensor_spec().tile(), expected.tile()) &&
           output.logical_shape() == expected.logical_shape() && output.memory_config() == expected.memory_config();
}

bool fill_supported(const Tensor& input) {
    if (!has_partial_tile(input)) {
        return false;
    }
    const auto dtype = input.dtype();
    if (dtype == DataType::BFLOAT8_B) {
        return false;
    }
    const auto elem = input.element_size();
    return elem == 2 || elem == 4;
}

// The gate must not depend on what happens to be allocated right now, or the same call would route differently
// across runs.
uint64_t static_l1_ceiling(const Tensor& input) {
    auto* device = input.device();
    const uint64_t size = device->l1_size_per_core();
    const uint64_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    return size > base ? size - base : 0;
}

uint64_t l1_output_reservation(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs) {
    if (attrs.fill_partial_tile) {
        return 0;
    }
    return get_pending_l1_output_reservation(
        input, attrs.output_logical_shape, attrs.output_mem_config, input.dtype(), input.layout());
}

uint64_t gate_l1_budget(const Tensor& input, uint64_t reserved) {
    const uint64_t ceiling = static_l1_ceiling(input);
    return ceiling > reserved ? ceiling - reserved : 0;
}

uint32_t tile_page_bytes_as(const Tensor& input, DataType dtype) {
    const auto format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
    return tt::round_up(input.tensor_spec().tile().get_tile_size(format), tt::tt_metal::hal::get_dram_alignment());
}

bool tile_fits_l1(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs, uint64_t reserved) {
    const uint64_t tile_bytes =
        tile_page_bytes_as(input, composes_via_bf16(input, attrs) ? DataType::BFLOAT16 : input.dtype());
    const uint64_t budget = gate_l1_budget(input, reserved);
    if (attrs.fill_partial_tile) {
        const uint64_t face_bytes = 16ull * 16ull * input.element_size();
        return 2 * tile_bytes + face_bytes <= budget;
    }
    return (static_cast<uint64_t>(cb_depth(attrs.read_batch, attrs.write_batch)) + 1) * tile_bytes <= budget;
}

// Smallest possible footprint (batch 1, depth 2) must fit; the factory scales batches down to it.
// Conditions that depend on neither the padding nor L1, so routing can decline before building params.
bool structurally_supported(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.buffer() == nullptr) {
        return false;
    }
    if (input.logical_shape().rank() != 4) {
        return false;
    }
    // ROW_MAJOR stays native: its device saving is smaller than the host cost of the codegen dispatch.
    if (input.layout() != ttnn::TILE_LAYOUT) {
        return false;
    }
    if (input.memory_config().is_sharded() || output_mem_config.is_sharded()) {
        return false;
    }
    return tile_geometry_ok(input) && dtype_ok(input);
}

bool padding_supported_and_fits(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs, uint64_t reserved) {
    if (attrs.fill_partial_tile) {
        return fill_supported(input) && tile_fits_l1(input, attrs, reserved);
    }
    const bool front_padded = attrs.front_n != 0 || attrs.front_c != 0 || attrs.front_h != 0 || attrs.front_w != 0;
    if (front_padded) {
        return false;
    }
    if (input.dtype() == DataType::BFLOAT8_B && has_partial_tile(input)) {
        return false;
    }
    if (has_partial_tile(input) && !fill_supported(input)) {
        return false;
    }
    return tile_fits_l1(input, attrs, reserved);
}

ttnn::prim::PadCodegenParams base_params(
    const Tensor& input,
    std::span<const uint32_t> front,
    std::span<const uint32_t> after,
    float value,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    const auto& shape = input.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && front.size() == 4 && after.size() == 4,
        "ttnn.pad: codegen path expects a rank-4 input and rank-4 padding");

    std::array<uint32_t, 4> out{};
    for (size_t i = 0; i < 4; ++i) {
        out[i] = front[i] + shape[i] + after[i];
    }

    ttnn::prim::PadCodegenParams p;
    p.output_logical_shape = ttnn::Shape(out);
    p.N_out = out[0];
    p.C_out = out[1];
    p.H_out = out[2];
    p.W_out = out[3];
    p.front_n = front[0];
    p.front_c = front[1];
    p.front_h = front[2];
    p.front_w = front[3];
    p.packed_pad_value = pack_pad_value(input.dtype(), value);
    p.read_batch = kReadBatch;
    p.write_batch = kWriteBatch;
    p.output_mem_config = output_mem_config;
    return p;
}

}  // namespace

uint32_t tile_page_bytes(const Tensor& input) { return tile_page_bytes_as(input, input.dtype()); }

bool composes_via_bf16(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs) {
    return input.dtype() == DataType::BFLOAT8_B && attrs.packed_pad_value != 0;
}

uint32_t pack_pad_value(DataType dtype, float value) {
    switch (dtype) {
        case DataType::FLOAT32: return std::bit_cast<uint32_t>(value);
        case DataType::INT32:
        case DataType::UINT32: {
            constexpr double kLimit = 4611686018427387904.0;  // keeps the int64 cast defined
            const double clamped = std::clamp(std::trunc(static_cast<double>(value)), -kLimit, kLimit);
            return static_cast<uint32_t>(static_cast<uint64_t>(static_cast<int64_t>(clamped)) & 0xFFFFFFFFull);
        }
        default: {
            const uint32_t half = bf16_bits(value);
            return (half << 16) | half;
        }
    }
}

bool has_partial_tile(const Tensor& input) {
    if (input.layout() != ttnn::TILE_LAYOUT) {
        return false;
    }
    const auto tile = input.tensor_spec().tile();
    const auto& shape = input.logical_shape();
    return shape[-2] % tile.get_height() != 0 || shape[-1] % tile.get_width() != 0;
}

ttnn::prim::PadCodegenParams make_codegen_params(
    const Tensor& input,
    std::span<const uint32_t> front,
    std::span<const uint32_t> after,
    float value,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    return base_params(input, front, after, value, output_mem_config);
}

std::optional<ttnn::prim::PadCodegenParams> plan_codegen(
    const Tensor& input,
    std::span<const uint32_t> front,
    std::span<const uint32_t> after,
    float value,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (!structurally_supported(input, output_mem_config)) {
        return std::nullopt;
    }
    auto p = base_params(input, front, after, value, output_mem_config);
    if (!padding_supported_and_fits(input, p, l1_output_reservation(input, p))) {
        return std::nullopt;
    }
    return p;
}

ttnn::prim::PadCodegenParams make_fill_partial_tile_params(const Tensor& input, float value) {
    const auto& shape = input.logical_shape();
    ttnn::prim::PadCodegenParams p;
    p.output_logical_shape = shape;
    p.N_out = shape[0];
    p.C_out = shape[1];
    p.H_out = shape[2];
    p.W_out = shape[3];
    p.packed_pad_value = pack_pad_value(input.dtype(), value);
    p.fill_partial_tile = true;
    p.output_mem_config = input.memory_config();
    return p;
}

tt::tt_metal::TensorSpec codegen_output_spec(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs) {
    if (attrs.fill_partial_tile) {
        return input.tensor_spec();
    }
    return tt::tt_metal::TensorSpec(
        attrs.output_logical_shape,
        tt::tt_metal::TensorLayout(
            input.dtype(),
            tt::tt_metal::PageConfig(input.layout(), input.tensor_spec().tile()),
            attrs.output_mem_config));
}

bool supported_by_codegen(
    const Tensor& input, const ttnn::prim::PadCodegenParams& attrs, const std::optional<Tensor>& optional_output) {
    if (!structurally_supported(input, attrs.output_mem_config) || attrs.output_logical_shape.rank() != 4) {
        return false;
    }
    if (optional_output.has_value() &&
        (optional_output->storage_type() != ttnn::StorageType::DEVICE || optional_output->buffer() == nullptr ||
         !output_matches(input, attrs, *optional_output))) {
        return false;
    }
    return padding_supported_and_fits(input, attrs, l1_output_reservation(input, attrs));
}

// Padding that stays inside the input's existing tiles: native copies without staging, codegen only adds launch cost.
bool tile_pad_adds_no_tiles(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs) {
    if (input.layout() != ttnn::TILE_LAYOUT) {
        return false;
    }
    const auto& s = input.logical_shape();
    const auto tile = input.tensor_spec().tile();
    return attrs.N_out == s[0] && attrs.C_out == s[1] &&
           tt::div_up(attrs.H_out, tile.get_height()) == tt::div_up(s[2], tile.get_height()) &&
           tt::div_up(attrs.W_out, tile.get_width()) == tt::div_up(s[3], tile.get_width());
}

bool is_demoted(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs) {
    if (attrs.fill_partial_tile) {
        return false;
    }
    return tile_pad_adds_no_tiles(input, attrs);
}

bool supported_execution_controls(bool use_multicore, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return use_multicore && !sub_core_grids.has_value();
}

}  // namespace ttnn::operations::data_movement::pad_codegen
