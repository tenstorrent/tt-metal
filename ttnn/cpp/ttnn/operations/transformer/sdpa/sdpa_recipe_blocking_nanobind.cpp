// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe_blocking_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <tt_stl/assert.hpp>

#include "sdpa.hpp"
#include "sdpa_recipe.hpp"
#include "sdpa_recipe_blocking.hpp"

namespace ttnn::operations::transformer {

namespace {
namespace recipe = sdpa::detail;

recipe::RecipeOp parse_op(const std::string& op) {
    if (op == "dense") {
        return recipe::RecipeOp::Dense;
    }
    if (op == "joint") {
        return recipe::RecipeOp::Joint;
    }
    if (op == "ring") {
        return recipe::RecipeOp::Ring;
    }
    if (op == "exp_ring") {
        return recipe::RecipeOp::ExpRing;
    }
    TT_THROW("Unknown SDPA recipe op '{}': expected dense, joint, ring or exp_ring", op);
}

recipe::PrecisionPolicy policy_of(ttnn::transformer::SDPAPrecision precision, DataType kv_type) {
    return recipe::resolve_precision_policy(recipe::select_recipe(precision, kv_type));
}

// (q_chunk_size, k_chunk_size, grid_x, grid_y, cost, jobs_per_core, l1_preferred, l1_minimum)
using BlockingTuple = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, double, uint32_t, uint64_t, uint64_t>;

BlockingTuple to_tuple(const recipe::RecipeBlocking& b) {
    return {
        b.q_chunk_size,
        b.k_chunk_size,
        static_cast<uint32_t>(b.grid.x),
        static_cast<uint32_t>(b.grid.y),
        b.cost,
        b.jobs_per_core,
        b.l1.preferred,
        b.l1.minimum};
}

recipe::RecipeBlockingProblem make_problem(
    const std::string& op,
    ttnn::transformer::SDPAPrecision precision,
    DataType kv_type,
    uint32_t batch,
    uint32_t q_heads,
    uint32_t q_rows,
    uint32_t k_rows,
    uint32_t head_dim,
    const CoreCoord& grid,
    uint64_t l1_bytes,
    uint32_t joint_q_rows,
    uint32_t joint_k_rows,
    uint32_t ring_size,
    uint32_t max_cores_per_head_batch,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    bool exp_mux_on_bottom_row) {
    recipe::RecipeBlockingProblem problem;
    problem.op = parse_op(op);
    problem.policy = policy_of(precision, kv_type);
    problem.batch = batch;
    problem.q_heads = q_heads;
    problem.q_rows = q_rows;
    problem.k_rows = k_rows;
    problem.joint_q_rows = joint_q_rows;
    problem.joint_k_rows = joint_k_rows;
    problem.ring_size = ring_size;
    problem.d_tiles = head_dim / 32;
    problem.grid = grid;
    problem.max_cores_per_head_batch = max_cores_per_head_batch;
    problem.l1_bytes = l1_bytes;
    problem.fixed_q_tiles = q_chunk_size / 32;
    problem.fixed_k_tiles = k_chunk_size / 32;
    problem.exp_mux_on_bottom_row = exp_mux_on_bottom_row;
    return problem;
}

#define SDPA_RECIPE_BLOCKING_ARGS                                                                               \
    nb::arg("op"), nb::arg("precision"), nb::arg("kv_dtype"), nb::arg("batch"), nb::arg("q_heads"),            \
        nb::arg("q_rows"), nb::arg("k_rows"), nb::arg("head_dim"), nb::arg("grid"), nb::arg("l1_bytes"),       \
        nb::kw_only(), nb::arg("joint_q_rows") = 0, nb::arg("joint_k_rows") = 0, nb::arg("ring_size") = 1,     \
        nb::arg("max_cores_per_head_batch") = 16, nb::arg("q_chunk_size") = 0, nb::arg("k_chunk_size") = 0, \
        nb::arg("exp_mux_on_bottom_row") = false

}  // namespace

void bind_sdpa_recipe_blocking(nb::module_& mod) {
    mod.def(
        "_sdpa_recipe_blocking",
        [](const std::string& op,
           ttnn::transformer::SDPAPrecision precision,
           DataType kv_type,
           uint32_t batch,
           uint32_t q_heads,
           uint32_t q_rows,
           uint32_t k_rows,
           uint32_t head_dim,
           const CoreCoord& grid,
           uint64_t l1_bytes,
           uint32_t joint_q_rows,
           uint32_t joint_k_rows,
           uint32_t ring_size,
           uint32_t max_cores_per_head_batch,
           uint32_t q_chunk_size,
           uint32_t k_chunk_size,
           bool exp_mux_on_bottom_row) -> std::optional<BlockingTuple> {
            const auto choice = recipe::choose_recipe_blocking(make_problem(
                op,
                precision,
                kv_type,
                batch,
                q_heads,
                q_rows,
                k_rows,
                head_dim,
                grid,
                l1_bytes,
                joint_q_rows,
                joint_k_rows,
                ring_size,
                max_cores_per_head_batch,
                q_chunk_size,
                k_chunk_size,
                exp_mux_on_bottom_row));
            if (!choice) {
                return std::nullopt;
            }
            return to_tuple(*choice);
        },
        SDPA_RECIPE_BLOCKING_ARGS,
        "Blocking the op would choose: (q_chunk, k_chunk, grid_x, grid_y, cost, jobs_per_core, l1_preferred, "
        "l1_minimum), or None.");
    mod.def(
        "_sdpa_recipe_blocking_candidates",
        [](const std::string& op,
           ttnn::transformer::SDPAPrecision precision,
           DataType kv_type,
           uint32_t batch,
           uint32_t q_heads,
           uint32_t q_rows,
           uint32_t k_rows,
           uint32_t head_dim,
           const CoreCoord& grid,
           uint64_t l1_bytes,
           uint32_t joint_q_rows,
           uint32_t joint_k_rows,
           uint32_t ring_size,
           uint32_t max_cores_per_head_batch,
           uint32_t q_chunk_size,
           uint32_t k_chunk_size,
           bool exp_mux_on_bottom_row) {
            std::vector<BlockingTuple> result;
            for (const auto& candidate : recipe::recipe_blocking_candidates(make_problem(
                     op,
                     precision,
                     kv_type,
                     batch,
                     q_heads,
                     q_rows,
                     k_rows,
                     head_dim,
                     grid,
                     l1_bytes,
                     joint_q_rows,
                     joint_k_rows,
                     ring_size,
                     max_cores_per_head_batch,
                     q_chunk_size,
                     k_chunk_size,
                     exp_mux_on_bottom_row))) {
                result.push_back(to_tuple(candidate));
            }
            return result;
        },
        SDPA_RECIPE_BLOCKING_ARGS,
        "Every feasible blocking, cheapest modeled cost first (same tuples as _sdpa_recipe_blocking).");
    mod.def(
        "_sdpa_recipe_geometry_rejection",
        [](const std::string& op,
           ttnn::transformer::SDPAPrecision precision,
           DataType kv_type,
           uint32_t q_chunk_size,
           uint32_t k_chunk_size,
           uint32_t head_dim) {
            return recipe::recipe_geometry_rejection(
                parse_op(op), policy_of(precision, kv_type), q_chunk_size / 32, k_chunk_size / 32, head_dim / 32);
        },
        nb::arg("op"),
        nb::arg("precision"),
        nb::arg("kv_dtype"),
        nb::arg("q_chunk_size"),
        nb::arg("k_chunk_size"),
        nb::arg("head_dim"),
        "Why a recipe geometry is unsupported, or None if it is supported.");
    mod.def(
        "_sdpa_recipe_l1_bytes",
        [](const std::string& op,
           ttnn::transformer::SDPAPrecision precision,
           DataType kv_type,
           uint32_t q_chunk_size,
           uint32_t k_chunk_size,
           uint32_t head_dim,
           uint32_t q_blocks_per_worker,
           uint32_t passes) {
            const auto estimate = recipe::recipe_l1_bytes(
                parse_op(op),
                policy_of(precision, kv_type),
                q_chunk_size / 32,
                k_chunk_size / 32,
                head_dim / 32,
                recipe::RecipeL1Context{.q_blocks_per_worker = q_blocks_per_worker, .passes = passes});
            return std::make_tuple(estimate.preferred, estimate.minimum);
        },
        nb::arg("op"),
        nb::arg("precision"),
        nb::arg("kv_dtype"),
        nb::arg("q_chunk_size"),
        nb::arg("k_chunk_size"),
        nb::arg("head_dim"),
        nb::kw_only(),
        nb::arg("q_blocks_per_worker") = 2,
        nb::arg("passes") = 1,
        "(preferred, minimum) circular-buffer bytes per core for a supported recipe geometry.");
}

}  // namespace ttnn::operations::transformer
