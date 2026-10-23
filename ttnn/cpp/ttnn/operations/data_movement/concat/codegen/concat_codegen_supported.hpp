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

inline constexpr int kLastAxis = -1;

// Most inputs the readers take. N-way means three or more inputs; two inputs use the dedicated two-input readers. The
// cap is an empirical Blackhole bound, not a hardware limit: larger input counts were not exercised.
inline constexpr uint32_t kConcatMaxNwayInputs = 8;

struct BatchPlan {
    uint32_t read_batch = 1;
    uint32_t write_batch = 1;
};

struct CbPages {
    uint32_t out_page = 0;
    uint32_t scratch_page = 0;
};

uint32_t cb_depth(uint32_t read_batch, uint32_t write_batch);

bool concats_width(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);
bool has_two_inputs(const std::vector<Tensor>& input_tensors);

CbPages cb_pages(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

uint64_t cb_footprint_bytes(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, const BatchPlan& batches);

// nullopt when even one page per batch does not fit `budget`.
std::optional<BatchPlan> plan_batches(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, uint64_t budget);

// Live L1 as the plan's budget, shared by routing, the hash and the factory. `output_allocated` says the output already
// sits below the live frontier; otherwise its L1 reservation is subtracted. The gate uses the static window instead.
std::optional<BatchPlan> plan_batches_live(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs, bool output_allocated);

// Routing-only: whether plan_batches_live finds a plan.
bool fits_live_l1(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

tt::tt_metal::TensorSpec codegen_output_spec(
    const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

// Budgets the static L1 window so routing and validate agree whatever is allocated at the time.
bool supported_by_codegen(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

// Routing-only perf gate; forced codegen never consults it. A case is demoted when codegen is correct but measured
// slower than native: on Wormhole any width concat, elsewhere a row-major width concat of three or more inputs with few
// output sticks.
bool is_demoted(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

// False when groups or sub_core_grids ask for something the codegen path does not implement.
bool supported_execution_controls(unsigned int groups, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::concat_codegen
