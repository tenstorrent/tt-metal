// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt_stl/reflection.hpp>

#include "groupnorm_sc_N_1_HW_C_b_device_operation_types.hpp"
#include "kernels/groupnorm_sc_N_1_HW_C_b_args.hpp"

// The C++ form of the Python planner (plan_or_raise, _choose_gx, _plan_sharded, _distribute, footprint_bytes).
// Integers in, integers out; every refusal is PlanNotImplemented with the Python's message.
namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b::planner {

// Page formats (Python Formats): input = output dtype, the affine dtype (bf16 when there is no affine) and layout.
struct Formats {
    tt::tt_metal::DataType in_dtype = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::DataType aff_dtype = tt::tt_metal::DataType::BFLOAT16;
    bool aff_tile = false;
    bool operator==(const Formats&) const = default;
};

// Everything the solve reads. Device state enters only as l1_budget (bytes for the static CBs, see l1_cb_budget).
struct PlanInputs {
    uint32_t N = 0, HW = 0, C = 0, G = 0;
    bool is_rm = false;
    bool has_gamma = false;
    bool has_beta = false;
    bool in_place = false;
    Formats fmt;
    uint32_t grid_x = 0, grid_y = 0;  // device->compute_with_storage_grid_size()
    uint32_t l1_budget = 0;
    std::optional<int32_t> max_resident_chunks;
    std::optional<int32_t> chunk_tiles_target;
    std::optional<int32_t> max_cores;
    // Block-sharded input only (sharded == false: all zero).
    bool sharded = false;
    bool shard_in_l1 = false;
    bool shard_row_major = false;
    uint32_t shard_grid_num_cores = 0;
    uint32_t sh = 0, sw = 0;                  // shard shape
    uint32_t x0 = 0, y0 = 0, gx = 0, gy = 0;  // shard grid bounding box
    bool operator==(const PlanInputs&) const = default;
};

// One core's work (Python CoreWork). Coordinates are logical.
struct CoreWork {
    uint32_t core_x = 0, core_y = 0;
    uint32_t unit_start = 0, num_blocks = 0, row_start = 0, rows_core = 0;
    uint32_t is_root = 0, slot = 0;
    uint32_t root_x = 0, root_y = 0;
    uint32_t xs = 0, vc = 0, l0 = 0, off0 = 0, g_unit = 0;
    uint32_t n_img = 1, n_seg = 1, seg_img0 = 0;
    std::vector<uint32_t> ops;  // op words (args header OP_*)
    // (shard stick, position in its tile row, sticks) per staged row
    std::vector<std::array<uint32_t, gn_sc_args::STAGED_WORDS>> staged;
    uint32_t rm_credit_pages = 0;
    std::vector<uint32_t> pilot_cols;  // logical x of the unit's columns
    uint32_t w_valid = 0;
};

// Python GroupNormPlan, scalars only. Every field is structural (fixes CB sizes, CT args, the core set) and goes into
// the program hash through plan_hash. The per-core work split follows from it and the inputs (build_work).
struct Plan {
    uint32_t N = 0, HW = 0, C = 0, G = 0, Ht = 0, Wt = 0, Cg = 0, span_tiles = 0, gps = 0, S = 0, gx = 0, W = 0;
    uint32_t G_core = 0, U = 0, P = 0, cpi = 0, h = 0, T = 0, R = 0, rows_core_max = 0, B_max = 0;
    uint32_t K_alloc = 0;
    bool stream = false, is_rm = false, has_gamma = false, has_beta = false;
    uint32_t grid_x = 0, grid_y = 0;
    uint32_t out_writer_chunks = 1;
    bool sharded = false, in_place = false;
    uint32_t map_period = 0, vc_last = 0;
    uint32_t shard_x0 = 0, shard_y0 = 0, shard_gx = 0, shard_gy = 0;  // sharded: the shard grid rectangle
    uint32_t NU = 1, NSEG = 1;
    bool ops_path = false;
    uint32_t stage_pages = 0;
    bool col_mask = false, mcast_row_major = false;
    uint32_t ring_pages = 0, pilot_cols = 1;
    Formats fmt;
};

// Bytes for the program's static CBs on every worker core: from the L1 allocator base to the lowest live L1 buffer
// (Python l1_cb_budget).
uint32_t l1_cb_budget(tt::tt_metal::IDevice* device);

// The plan's inputs for a call (reads the device's free L1). Throws std::invalid_argument / refusals never: the
// support contract is checked before.
PlanInputs plan_inputs(const GroupnormScN1HwCbParams& attrs, const GroupnormScN1HwCbInputs& tensors);

// The scalar solve, memoized on its inputs (process-wide; the plan is host data). Throws PlanNotImplemented. It
// builds no per-core data, so compute_program_hash can run it on every call.
const Plan& solve_plan(const PlanInputs& inputs);

// The per-core work split of a solved plan (Python order: _distribute (interleaved) / grid rows then columns
// (sharded)): per-core ops, staged rows and pilot columns. Only a cache miss (create_descriptor) builds it.
std::vector<CoreWork> build_work(const PlanInputs& inputs, const Plan& plan);

// Hash of every structural scalar of the plan, for compute_program_hash.
ttsl::hash::hash_t plan_hash(const Plan& plan);

// Python footprint/staging helpers shared by the planner and the factory's CB sizing.
uint32_t input_ring_pages(uint32_t T, uint32_t K_alloc, bool stream, bool is_rm);
uint32_t staging_bytes(const Plan& plan);

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b::planner
