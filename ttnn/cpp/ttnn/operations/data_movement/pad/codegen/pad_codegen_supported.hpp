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

// Terms used below:
//   gate       a correctness or device-resource check; an input that fails it cannot run on codegen.
//   demotion   the input passes the gate and codegen is correct, but slower than native, so routing picks native.
//   static L1 budget  the L1 bank minus the pending output reservation; does not depend on what is allocated now,
//                     so the gate gives the same answer at routing and at the prim's validate.
//   live L1 budget    the L1 free on the device right now (get_max_l1_space); moves with every allocation.

// Output CB depth in pages for the given batching.
uint32_t cb_depth(uint32_t read_batch, uint32_t write_batch);

// Bytes of one TILE page of `dtype`, rounded up to the DRAM alignment.
uint32_t dram_aligned_tile_bytes(const Tensor& input, tt::tt_metal::DataType dtype);

// Page batching of the tile pad's reader and writer.
struct BatchPlan {
    uint32_t read_batch = 1;
    uint32_t write_batch = 1;
};

// Per-core CB bytes of the program for `attrs` at `batches`. The single footprint both the routing gate and the
// program factory plan against.
uint64_t cb_footprint_bytes(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs, const BatchPlan& batches);

// Largest batching whose footprint fits `budget`, halving from the default down to one page each; nullopt when even
// that does not fit.
std::optional<BatchPlan> plan_batches(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs, uint64_t budget);

// plan_batches against the live L1 budget. The program hash and the program factory call it after the output is
// allocated, so the live query already accounts for it. The gate budgets the static window instead, so a live read
// there would change the answer between routing and validate.
std::optional<BatchPlan> plan_batches_live(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// Routing-only: whether the smallest CB plan fits the live L1 budget once the outputs routing has not allocated yet
// are reserved. Auto routing declines to native when it does not; forced codegen skips it and raises in the program
// factory instead.
bool fits_live_l1(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// A nonzero fill of a block-float tensor has no scalar page pattern, so the caller pads a bfloat16 copy instead.
bool composes_via_bf16(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// The 32-bit word the kernels replicate into pad regions.
uint32_t pack_pad_value(tt::tt_metal::DataType dtype, float value);

// Builds the attributes for padding a rank-4 tensor by `after` trailing elements per dim. The kernels pad only at
// the end, so there is no leading padding to carry.
ttnn::prim::PadCodegenParams make_codegen_params(
    const Tensor& input,
    std::span<const uint32_t> after,
    float value,
    const tt::tt_metal::MemoryConfig& output_mem_config);

// Padding-independent part of the gate: structural support (device, arch, rank 4, layout, sharding, tile geometry,
// dtype) and no block-float partial tile.
bool in_codegen_scope(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

// Routing entry: the attributes when the gate accepts the call, nullopt when it declines. in_codegen_scope first so
// cheap rejections skip building params, then make_codegen_params and supported_by_codegen. The caller must already
// have declined nonzero leading padding.
std::optional<ttnn::prim::PadCodegenParams> plan_codegen(
    const Tensor& input,
    std::span<const uint32_t> after,
    float value,
    const tt::tt_metal::MemoryConfig& output_mem_config);

// True when the TILE input has a partial border tile.
bool has_partial_tile(const Tensor& input);

// Output spec the codegen path builds; a caller-supplied output must match it.
tt::tt_metal::TensorSpec codegen_output_spec(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// The gate for already-built attributes, against the static L1 budget; the codegen prim's validation calls it and
// plan_codegen delegates to it.
bool supported_by_codegen(
    const Tensor& input,
    const ttnn::prim::PadCodegenParams& attrs,
    const std::optional<Tensor>& optional_output = std::nullopt);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only.
bool is_demoted(const Tensor& input, const ttnn::prim::PadCodegenParams& attrs);

// False when use_multicore or sub_core_grids ask for something the codegen path does not implement.
bool supported_execution_controls(bool use_multicore, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::pad_codegen
