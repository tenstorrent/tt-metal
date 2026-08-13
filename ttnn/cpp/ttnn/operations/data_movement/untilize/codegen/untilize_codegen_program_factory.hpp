// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// _CB_IN_ID / _CB_OUT_ID from ops/untilize/spec.py.
inline constexpr uint32_t kUntilizeCbInId = 0;
inline constexpr uint32_t kUntilizeCbOutId = 16;

// Mirrors common/codegen_common/builder_utils.USABLE_L1 /
// factory/cb_policy._DEFAULT_USABLE_L1 -- the CB-fit budget both the factory
// and supported_by_codegen()'s gate must agree on so they never diverge.
inline constexpr uint32_t kUntilizeUsableL1 = 1'400'000;

struct UntilizeCodegenParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct UntilizeCodegenInputs {
    Tensor input;
};

// Which branch of ops/untilize/spec.py::build_untilize_tile (or the
// with-unpadding builder) this input reaches. Computed once by
// plan_untilize_dispatch() and consumed identically by the factory (to build
// the right descriptor) and by supported_by_codegen() (to know which
// pages-per-unit value bounds the CB-fit gate for *this* input) -- so the two
// can never diverge on which branch fires, which is the failure mode the
// porting guide calls out for this op (a guard written for one branch leaving
// another branch's CB size unbounded).
enum class UntilizeDispatchKind { ColumnParallel, TwoDColumn, Cliff, Single, WithUnpadding };

struct UntilizeDispatchPlan {
    UntilizeDispatchKind kind = UntilizeDispatchKind::Single;
    // Physical (tile-aligned) geometry. For WithUnpadding these are derived
    // from the padded shape (ceil-divided), matching spec.py's
    // build_untilize_with_unpadding; for the other kinds they are the exact
    // (already tile-aligned) tensor tile geometry.
    uint32_t total_tile_rows = 0;
    uint32_t Wt = 0;
    // Pages-per-unit (P) driving plan_cb_depths() for whichever branch fires:
    // Wt for Single/Cliff/WithUnpadding, max(tpc1, tpc2) for ColumnParallel,
    // tpc = Wt / ncol for TwoDColumn.
    uint32_t pages_per_unit = 0;
    // TwoDColumn only.
    uint32_t ncol = 1;
    bool fp32_dest_acc = false;
    uint32_t tile_size_bytes = 0;  // max(input tile size, output tile size)
};

// Transliterates spec.py::build_untilize_tile's branch selection (plus the
// with-unpadding entry condition from untilize.cpp's DRAM+padding redirect)
// against `input`'s *actual* device grid, so the branch this returns is
// exactly the branch the factory will build.
UntilizeDispatchPlan plan_untilize_dispatch(const Tensor& input, bool with_unpadding);

// True iff `plan`'s pages_per_unit fits in `budget` bytes of L1 under
// plan_cb_depths()'s single-buffered (tier-3) floor: 2 * P * tile_size <=
// budget. This is the floor every branch must clear -- if even this fails,
// ops/common/codegen_common/factory/cb_policy.py::plan_cb_depths raises
// rather than degrading further, so the C++ port has no fallback and must
// reject upstream in supported_by_codegen().
bool untilize_cb_plan_fits(const UntilizeDispatchPlan& plan, uint32_t budget_bytes);

struct UntilizeCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeCodegenParams& operation_attributes,
        const UntilizeCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
