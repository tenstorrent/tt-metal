// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

// Terms used below:
//   gate       a correctness or device-resource check; an input that fails it cannot run on codegen.
//   demotion   the input passes the gate and codegen is correct, but slower than native, so routing picks native.
//   static L1 budget  the L1 bank minus the pending output reservation; does not depend on what is allocated now,
//                     so the gate gives the same answer at routing and at the prim's validate.
//   live L1 budget    the L1 free on the device right now (get_max_l1_space); moves with every allocation.

// Index of the stick (innermost) dimension.
inline constexpr int kLastAxis = -1;

// Compile-time ceiling of the N-way readers.
inline constexpr uint32_t kConcatMaxNwayInputs = 8;

// Page batching of the reader and writer.
struct BatchPlan {
    uint32_t read_batch = 1;
    uint32_t write_batch = 1;
};

// Page strides of the circular buffers of the program for `attrs`.
struct CbPages {
    // Slot stride of the staging CB the reader fills and the writer drains.
    uint32_t out_page = 0;
    // Bytes of the scratch CB's single page; zero when the leaf needs none.
    uint32_t scratch_page = 0;
};

// Output CB depth in pages for the given batching.
uint32_t cb_depth(uint32_t read_batch, uint32_t write_batch);

// Which reader leaf the inputs map to.
bool concats_width(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);
bool is_two_input_leaf(const std::vector<Tensor>& input_tensors);

CbPages cb_pages(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

// Per-core CB bytes of the program for `attrs` at `batches`. The single footprint both the routing gate and the
// program factory plan against.
uint64_t cb_footprint_bytes(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, const BatchPlan& batches);

// Largest batching whose footprint fits `budget`, halving from the default down to one page each; nullopt when even
// that does not fit.
std::optional<BatchPlan> plan_batches(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, uint64_t budget);

// plan_batches against the live L1 budget. The program hash and the program factory call it after the output is
// allocated, so the live query already accounts for it. The gate budgets the static window instead, so a live read
// there would change the answer between routing and validate.
std::optional<BatchPlan> plan_batches_live(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

// Routing-only: whether the smallest CB plan fits the live L1 budget once the output, which routing has not allocated
// yet, is reserved. Auto routing declines to native when it does not; forced codegen skips it and raises in the
// program factory instead.
bool fits_live_l1(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

// Output spec the codegen path builds; a caller-supplied output must match it.
tt::tt_metal::TensorSpec codegen_output_spec(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

// The correctness gate shared by routing and the codegen prim's validation step, against the static L1 budget.
bool supported_by_codegen(
    const std::vector<Tensor>& input_tensors,
    const ttnn::prim::ConcatCodegenParams& attrs,
    const std::optional<Tensor>& optional_output = std::nullopt);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only; forced codegen never consults it.
bool is_demoted(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

// False when groups or sub_core_grids ask for something the codegen path does not implement.
bool supported_execution_controls(unsigned int groups, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::concat_codegen
