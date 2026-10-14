// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "untilize_codegen_device_operation.hpp"

namespace ttnn::prim::untilize_codegen_detail {

// The three CB depth tiers a codegen builder can plan, plus `Native`: no codegen plan fits the L1
// budget it was given. The codegen op never serves `Native` itself -- ttnn::untilize consults the
// same chooser before dispatching (codegen_cb_plan_fits_live_l1) and routes such a case to the
// native untilize prim as a whole, so inside the codegen op `Native` is a hard error.
enum class CodegenCbPlan : uint8_t { DoubleBoth, DoubleIn, SingleBoth, Native };

struct CbPlan {
    uint32_t cb_in_depth;
    uint32_t cb_out_depth;
    uint32_t read_batch;
};

std::optional<CbPlan> plan_cb_depths(
    uint64_t usable_l1, uint32_t pages_per_unit, uint32_t page_size, uint32_t block_units);

uint32_t compute_block_ct_dim(uint32_t wt, bool fp32);
uint32_t choose_2d_ncol(uint32_t total_tile_rows, uint32_t wt, uint32_t valid_cores);

struct ChosenCodegenCbPlan {
    CodegenCbPlan tier;
    std::optional<CbPlan> depths;
};

// Live-L1 CB tier for this dispatch: the L1 free right now (get_max_l1_space), less
// `reserved_l1_bytes_per_core`, planned exactly as the program factory's builders plan it. Output
// tile size comes from UntilizeCodegenDeviceOperation::compute_output_specs so bf8_b->bf16
// demotion is not copied.
//
// Two callers, two moments:
//   * compute_program_hash / create_descriptor pass 0: they run after the op's output tensor has
//     been allocated, so the live query already reflects it.
//   * ttnn::untilize's routing runs BEFORE the output is allocated and passes the output's
//     pending per-core L1 footprint (get_pending_l1_output_reservation), so its Native-vs-codegen
//     decision predicts the budget the two later calls will actually see.
ChosenCodegenCbPlan choose_codegen_cb_plan(
    const UntilizeCodegenOperationAttributes& attrs,
    const UntilizeCodegenTensorArgs& tensor_args,
    uint32_t reserved_l1_bytes_per_core = 0);

}  // namespace ttnn::prim::untilize_codegen_detail
