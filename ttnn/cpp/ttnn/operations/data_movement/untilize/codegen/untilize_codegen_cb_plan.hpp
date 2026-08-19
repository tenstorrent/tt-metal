// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "untilize_codegen_device_operation.hpp"

namespace ttnn::prim::untilize_codegen_detail {

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

// Live-L1 CB tier for this (already output-allocated) dispatch. Output tile size comes from
// UntilizeCodegenDeviceOperation::compute_output_specs so bf8_b->bf16 demotion is not copied.
ChosenCodegenCbPlan choose_codegen_cb_plan(
    const UntilizeCodegenOperationAttributes& attrs, const UntilizeCodegenTensorArgs& tensor_args);

// Discrete native-fallback identity hashed with the codegen tier. Zeros when tier is not Native
// or when enough_space_height is true. split_valid is false when the block split has no solution
// (sentinel: forces a cache miss so create_descriptor can TT_FATAL as the factory does today).
struct NativeCacheIdentity {
    bool enough_space_height = false;
    bool split_valid = true;
    uint32_t ncores = 0;
    uint32_t nblocks_per_core = 0;
    uint32_t single_block_size = 0;
    uint32_t single_block_size_cliff_row = 0;
    uint32_t single_block_size_cliff_col = 0;
    bool has_cliff_row = false;
    bool has_cliff_col = false;
    uint32_t full_cores_per_row = 0;
    uint32_t full_cores_per_col = 0;
    uint32_t single_sub_block_size = 0;
};

NativeCacheIdentity native_cache_identity(
    const UntilizeCodegenOperationAttributes& attrs,
    const UntilizeCodegenTensorArgs& tensor_args,
    CodegenCbPlan plan);

}  // namespace ttnn::prim::untilize_codegen_detail
