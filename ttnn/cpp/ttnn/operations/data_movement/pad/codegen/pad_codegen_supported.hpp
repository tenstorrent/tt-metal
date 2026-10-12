// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::pad_codegen {

// Output CB depth in pages for the given batching.
uint32_t cb_depth(uint32_t read_batch, uint32_t write_batch);

// Aligned bytes of one TILE page.
uint32_t tile_page_bytes(const Tensor& input);

// Pages of the fill pass's tile scratch CB.
inline constexpr uint32_t kFillScratchPages = 2;

// Bytes of the fill pass's pad-value CB: one face.
uint32_t fill_face_bytes(const Tensor& input);

// Page batching of the tile pad's reader and writer.
struct BatchPlan {
    uint32_t read_batch = 1;
    uint32_t write_batch = 1;
};

// Per-core CB bytes of the program for `attrs` at `batches` (the fill pass has no batching). The single footprint
// both the routing gate and the program factory plan against.
uint64_t cb_footprint_bytes(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs, const BatchPlan& batches);

// Largest batching whose footprint fits `budget`, halving from the default down to one page each; nullopt when even
// that does not fit.
std::optional<BatchPlan> plan_batches(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs, uint64_t budget);

// plan_batches against the L1 free on the device right now. Only the program hash and the program factory call it:
// both run after the output is allocated, so the live query already accounts for it. The routing gate budgets the
// static window instead, so a live read there would change the answer between routing and validate.
std::optional<BatchPlan> plan_batches_live(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// A nonzero fill of a block-float tensor has no scalar page pattern, so the caller pads a bfloat16 copy instead.
bool composes_via_bf16(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// The 32-bit word the kernels replicate into pad regions.
uint32_t pack_pad_value(tt::tt_metal::DataType dtype, float value);

// Builds the attributes for padding a rank-4 tensor by `front`/`after` elements per dim.
ttnn::prim::PadCodegenParams make_codegen_params(
    const Tensor& input,
    std::span<const uint32_t> front,
    std::span<const uint32_t> after,
    float value,
    const tt::tt_metal::MemoryConfig& output_mem_config);

// Routing entry: the attributes when the codegen gate accepts the call, nullopt when it declines. Same decision as
// make_codegen_params followed by supported_by_codegen, but declines before the L1 query when it can.
std::optional<ttnn::prim::PadCodegenParams> plan_codegen(
    const Tensor& input,
    std::span<const uint32_t> front,
    std::span<const uint32_t> after,
    float value,
    const tt::tt_metal::MemoryConfig& output_mem_config);

// Attributes for the in-place pass that fills the unused lanes of a TILE input's partial border tiles.
ttnn::prim::PadCodegenParams make_fill_partial_tile_params(const Tensor& input, float value);

// True when the input has a partial border tile.
bool has_partial_tile(const Tensor& input);

// Output spec the codegen path builds; a caller-supplied output must match it.
tt::tt_metal::TensorSpec codegen_output_spec(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// Correctness and device-resource gate, shared by the free function's routing, pad_force_codegen and the
// codegen prim's checks.
bool supported_by_codegen(
    const Tensor& input,
    const ttnn::prim::PadCodegenParams& attrs,
    const std::optional<Tensor>& optional_output = std::nullopt);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only.
bool is_demoted(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// False when use_multicore or sub_core_grids ask for something the codegen path does not implement.
bool supported_execution_controls(bool use_multicore, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::pad_codegen
