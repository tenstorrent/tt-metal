// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_sc_N_1_HW_C_b_device_operation.hpp"

#include <stdexcept>
#include <string>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "groupnorm_sc_N_1_HW_C_b_planner.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b {

using namespace tt::tt_metal;

namespace {

constexpr const char* kOp = "groupnorm_sc_N_1_HW_C_b";

bool gn_is_affine_shape(const Tensor& t, uint32_t C) {
    const auto& s = t.logical_shape();
    return s.rank() == 4 && s[0] == 1 && s[1] == 1 && s[2] == 1 && s[3] == C;
}

// (grid_x, grid_y) of a sharded input's shard grid (Python _shard_placement), or nullopt.
std::optional<std::pair<uint32_t, uint32_t>> gn_shard_placement(const Tensor& input) {
    const auto& mc = input.memory_config();
    if (!mc.is_sharded() || !mc.shard_spec().has_value()) {
        return std::nullopt;
    }
    const auto bbox = mc.shard_spec()->grid.bounding_box();
    return std::pair<uint32_t, uint32_t>{
        static_cast<uint32_t>(bbox.end_coord.x - bbox.start_coord.x + 1),
        static_cast<uint32_t>(bbox.end_coord.y - bbox.start_coord.y + 1)};
}

std::string gn_alignment(uint32_t HW, uint32_t C) {
    if (HW % tt::constants::TILE_HEIGHT == 0 && C % tt::constants::TILE_WIDTH == 0) {
        return "tile_aligned";
    }
    if (C % tt::constants::TILE_WIDTH != 0) {
        return "c_non_aligned";
    }
    return "hw_non_aligned";
}

// Python tag_shard_fit; RM shards are granular in 1 x 8 elements.
std::string gn_shard_fit(
    const std::optional<std::pair<uint32_t, uint32_t>>& placement, uint32_t rows, uint32_t C, bool tile) {
    if (!placement.has_value()) {
        return "none";
    }
    constexpr uint32_t kRmWidthGrain = 8;
    const auto [gx, gy] = *placement;
    const uint32_t hg = tile ? tt::constants::TILE_HEIGHT : 1;
    const uint32_t wg = tile ? tt::constants::TILE_WIDTH : kRmWidthGrain;
    const uint32_t sh = tt::div_up(tt::div_up(rows, gy), hg) * hg;
    const uint32_t sw = tt::div_up(tt::div_up(C, gx), wg) * wg;
    if ((gy - 1) * sh >= rows || (gx - 1) * sw >= C) {
        return "illegal";
    }
    if (sh * gy == rows && sw * gx == C) {
        return "even";
    }
    return "ragged";
}

template <typename V>
void gn_refuse_unless(bool supported, std::string_view axis, const V& value, std::string_view allowed) {
    if (!supported) {
        throw UnsupportedAxisValue(fmt::format("{}: {}={} not in SUPPORTED [{}]", kOp, axis, value, allowed));
    }
}

void gn_exclude_if(bool hit, std::string_view cell) {
    if (hit) {
        throw ExcludedCell(fmt::format("{}: unsupported combination (refinement candidate): {}", kOp, cell));
    }
}

bool gn_is_supported_dtype(DataType d) {
    return d == DataType::BFLOAT16 || d == DataType::FLOAT32 || d == DataType::BFLOAT8_B;
}

// Same device and buffer type, same address: the output is (a view of) the input's buffer.
bool gn_shares_buffer(const Tensor& a, const Tensor& b) {
    const Buffer* x = a.buffer();
    const Buffer* y = b.buffer();
    return x != nullptr && y != nullptr && a.device() == b.device() && x->buffer_type() == y->buffer_type() &&
           x->address() == y->address();  // smuggled-rta-ok: aliasing check in validation, not a runtime arg
}

void gn_validate_output(const GroupnormScN1HwCbParams& attrs, const GroupnormScN1HwCbInputs& t) {
    if (!t.output.has_value()) {
        return;
    }
    const Tensor& out = *t.output;
    const TensorSpec expected = GroupnormScN1HwCbDeviceOperation::compute_output_specs(attrs, t);
    TT_FATAL(
        out.logical_shape() == expected.logical_shape(),
        "{}: output_tensor shape {} does not match {}",
        kOp,
        out.logical_shape(),
        expected.logical_shape());
    TT_FATAL(
        out.dtype() == expected.data_type(),
        "{}: output_tensor dtype {} does not match {}",
        kOp,
        out.dtype(),
        expected.data_type());
    TT_FATAL(
        out.layout() == expected.layout(),
        "{}: output_tensor layout {} does not match {}",
        kOp,
        out.layout(),
        expected.layout());
    TT_FATAL(
        out.memory_config() == expected.memory_config(),
        "{}: output_tensor memory_config {} does not match {}",
        kOp,
        out.memory_config(),
        expected.memory_config());
    TT_FATAL(
        out.tensor_spec() == expected, "{}: output_tensor spec {} does not match {}", kOp, out.tensor_spec(), expected);
    // In place is the sharded in_place path only: there every core reads each shard tile before it packs over it.
    // Out of place, an aliased output is refused (the interleaved pass 2 re-reads chunks the other share writes).
    TT_FATAL(
        attrs.in_place || !gn_shares_buffer(out, t.input),
        "{}: output_tensor shares the input's buffer; pass in_place=True (BLOCK_SHARDED) for an in-place call",
        kOp);
}

}  // namespace

void check_support(const GroupnormScN1HwCbParams& attrs, const GroupnormScN1HwCbInputs& t) {
    const Tensor& input = t.input;
    const auto& shape = input.logical_shape();
    // (1) Structural ValueErrors.
    if (shape.rank() != 4) {
        throw std::invalid_argument(
            fmt::format("{}: input rank must be 4 (N, 1, HW, C), got rank {}", kOp, shape.rank()));
    }
    if (shape[1] != 1) {
        throw std::invalid_argument(fmt::format("{}: input shape[1] must be 1, got shape {}", kOp, shape));
    }
    const uint32_t N = shape[0];
    const uint32_t HW = shape[2];
    const uint32_t C = shape[3];
    if (attrs.num_groups < 1 || C % attrs.num_groups != 0) {
        throw std::invalid_argument(
            fmt::format("{}: num_groups={} must be >= 1 and divide C={}", kOp, attrs.num_groups, C));
    }
    if (t.gamma.has_value() && !gn_is_affine_shape(*t.gamma, C)) {
        throw std::invalid_argument(
            fmt::format("{}: gamma must have shape (1, 1, 1, {}), got {}", kOp, C, t.gamma->logical_shape()));
    }
    if (t.beta.has_value() && !gn_is_affine_shape(*t.beta, C)) {
        throw std::invalid_argument(
            fmt::format("{}: beta must have shape (1, 1, 1, {}), got {}", kOp, C, t.beta->logical_shape()));
    }
    if (!(attrs.eps > 0.0f)) {
        throw std::invalid_argument(fmt::format("{}: eps must be > 0, got {}", kOp, attrs.eps));
    }
    const auto placement = gn_shard_placement(input);
    if (attrs.in_place && !placement.has_value()) {
        throw std::invalid_argument(fmt::format("{}: in_place=True requires a BLOCK_SHARDED (L1) input", kOp));
    }

    // (2) Axes.
    const bool has_g = t.gamma.has_value();
    const bool has_b = t.beta.has_value();
    const std::string affine = has_g && has_b ? "gamma_beta" : has_g ? "gamma_only" : has_b ? "beta_only" : "no_affine";
    const Tensor* ref = has_g ? &*t.gamma : has_b ? &*t.beta : nullptr;
    const bool tile = input.layout() == Layout::TILE;
    const std::string alignment = gn_alignment(HW, C);
    const TensorMemoryLayout mem_layout = input.memory_config().memory_layout();
    const std::string shard_fit = gn_shard_fit(placement, N * HW, C, tile);
    if (has_g && has_b && (t.gamma->dtype() != t.beta->dtype() || t.gamma->layout() != t.beta->layout())) {
        throw UnsupportedAxisValue(fmt::format(
            "{}: gamma and beta must share dtype and layout (gamma {}/{}, beta {}/{})",
            kOp,
            t.gamma->dtype(),
            t.gamma->layout(),
            t.beta->dtype(),
            t.beta->layout()));
    }

    // (3) SUPPORTED, in the Python's axis order.
    gn_refuse_unless(gn_is_supported_dtype(input.dtype()), "dtype", input.dtype(), "BFLOAT16, FLOAT32, BFLOAT8_B");
    gn_refuse_unless(tile || input.layout() == Layout::ROW_MAJOR, "layout", input.layout(), "TILE, ROW_MAJOR");
    gn_refuse_unless(affine != "beta_only", "affine", affine, "gamma_beta, gamma_only, no_affine");
    if (ref != nullptr) {
        gn_refuse_unless(
            gn_is_supported_dtype(ref->dtype()), "affine_dtype", ref->dtype(), "BFLOAT16, FLOAT32, BFLOAT8_B, none");
        gn_refuse_unless(
            ref->layout() == Layout::TILE || ref->layout() == Layout::ROW_MAJOR,
            "affine_layout",
            ref->layout(),
            "ROW_MAJOR, TILE, none");
    }
    const bool block_sharded = mem_layout == TensorMemoryLayout::BLOCK_SHARDED;
    gn_refuse_unless(
        mem_layout == TensorMemoryLayout::INTERLEAVED || block_sharded,
        "memory_layout",
        mem_layout,
        "INTERLEAVED, BLOCK_SHARDED");
    gn_refuse_unless(shard_fit != "illegal", "shard_fit", shard_fit, "none, even, ragged");

    // EXCLUSIONS, in the Python's order.
    if (block_sharded) {
        gn_exclude_if(input.dtype() == DataType::FLOAT32, "{memory_layout: BLOCK_SHARDED, dtype: float32}");
        gn_exclude_if(
            ref != nullptr && ref->dtype() == DataType::FLOAT32,
            "{memory_layout: BLOCK_SHARDED, affine_dtype: float32}");
        gn_exclude_if(
            ref != nullptr && ref->layout() == Layout::TILE, "{memory_layout: BLOCK_SHARDED, affine_layout: TILE}");
        gn_exclude_if(input.dtype() == DataType::BFLOAT8_B, "{memory_layout: BLOCK_SHARDED, dtype: bfloat8_b}");
        gn_exclude_if(
            ref != nullptr && ref->dtype() == DataType::BFLOAT8_B,
            "{memory_layout: BLOCK_SHARDED, affine_dtype: bfloat8_b}");
        gn_exclude_if(alignment == "hw_non_aligned", "{memory_layout: BLOCK_SHARDED, alignment: hw_non_aligned}");
        gn_exclude_if(alignment == "c_non_aligned", "{memory_layout: BLOCK_SHARDED, alignment: c_non_aligned}");
    }

    // Shape-class gate: the plan (throws PlanNotImplemented).
    static_cast<void>(planner::solve_plan(planner::plan_inputs(attrs, t)));
}

GroupnormScN1HwCbDeviceOperation::program_factory_t GroupnormScN1HwCbDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return GroupnormScN1HwCbProgramFactory{};
}

void GroupnormScN1HwCbDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    check_support(attrs, tensor_args);
    gn_validate_output(attrs, tensor_args);
}

void GroupnormScN1HwCbDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    gn_validate_output(attrs, tensor_args);
}

ttsl::hash::hash_t GroupnormScN1HwCbDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // eps is left out: it is a runtime scalar (compute EPS_BITS) that the factory's override_runtime_arguments
    // re-applies on every hit. Everything structural is in: the tuning, the tensor specs, affine presence and the
    // solved plan (which holds the L1 budget only through the geometry it fixes). The output spec follows from the
    // input spec and in_place; a preallocated output must match it.
    const planner::Plan& plan = planner::solve_plan(planner::plan_inputs(attrs, tensor_args));
    std::optional<TensorSpec> gamma_spec;
    std::optional<TensorSpec> beta_spec;
    if (tensor_args.gamma.has_value()) {
        gamma_spec = tensor_args.gamma->tensor_spec();
    }
    if (tensor_args.beta.has_value()) {
        beta_spec = tensor_args.beta->tensor_spec();
    }
    return tt::tt_metal::operation::hash_operation<GroupnormScN1HwCbDeviceOperation>(
        attrs.num_groups,
        attrs.in_place,
        attrs.max_resident_chunks,
        attrs.chunk_tiles_target,
        attrs.max_cores,
        attrs.chunk_zones,
        tensor_args.input.tensor_spec(),
        gamma_spec,
        beta_spec,
        planner::plan_hash(plan));
}

GroupnormScN1HwCbDeviceOperation::spec_return_value_t GroupnormScN1HwCbDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    if (attrs.in_place) {
        return input.tensor_spec();
    }
    const bool sharded = input.memory_config().is_sharded();
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(
            input.dtype(),
            PageConfig(sharded ? input.layout() : Layout::TILE),
            sharded ? input.memory_config() : ttnn::DRAM_MEMORY_CONFIG));
}

GroupnormScN1HwCbDeviceOperation::tensor_return_value_t GroupnormScN1HwCbDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return *tensor_args.output;
    }
    if (attrs.in_place) {
        return tensor_args.input;
    }
    return ttnn::create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b

namespace ttnn::prim {

Tensor groupnorm_sc_N_1_HW_C_b(
    const Tensor& input,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    const std::optional<Tensor>& output,
    const ttnn::operations::groupnorm_sc_N_1_HW_C_b::GroupnormScN1HwCbParams& params) {
    using OperationType = ttnn::operations::groupnorm_sc_N_1_HW_C_b::GroupnormScN1HwCbDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.input = input, .gamma = gamma, .beta = beta, .output = output});
}

}  // namespace ttnn::prim
