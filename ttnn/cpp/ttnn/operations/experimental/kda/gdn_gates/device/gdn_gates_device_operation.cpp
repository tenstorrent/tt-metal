// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_gates_device_operation.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

void check_input(const Tensor& t, const char* name) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr,
        "gdn_gates: {} must be an allocated device tensor",
        name);
    TT_FATAL(t.layout() == Layout::TILE, "gdn_gates: {} must use TILE layout", name);
    TT_FATAL(!t.is_sharded(), "gdn_gates: {} must use interleaved memory", name);
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "gdn_gates: {} must be BFLOAT16", name);
}

}  // namespace

GdnGatesOperation::program_factory_t GdnGatesOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GdnGatesProgramFactory{};
}

void GdnGatesOperation::validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_input(in.gab, "gab");
    check_input(in.dt_bias, "dt_bias");
    check_input(in.a_neg, "a_neg");
    TT_FATAL(
        in.gab.device() == in.dt_bias.device() && in.gab.device() == in.a_neg.device(),
        "gdn_gates: inputs must be on the same device");
    TT_FATAL(!attrs.output_mem_config.is_sharded(), "gdn_gates: output memory configuration must be interleaved");
    TT_FATAL(attrs.num_heads > 0 && attrs.num_heads <= tt::constants::TILE_WIDTH, "gdn_gates: num_heads must be 1..32");
    TT_FATAL(std::isfinite(attrs.beta_scale), "gdn_gates: beta_scale must be finite");
    const auto& s = in.gab.logical_shape();
    TT_FATAL(
        (s.rank() == 3 || s.rank() == 4) && s.rank() == attrs.rank && s.volume() / (s[-1] * s[-2]) == 1,
        "gdn_gates: gab must be [1,T,W] or [1,1,T,W]");
    TT_FATAL(
        attrs.sequence > 0 && attrs.sequence % tt::constants::TILE_HEIGHT == 0 && s[-2] == attrs.sequence,
        "gdn_gates: T must be positive and tile aligned");
    const uint32_t row_tiles = in.gab.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(
        attrs.a_col_tile < row_tiles && attrs.b_col_tile < row_tiles,
        "gdn_gates: a/b column offset outside gab (row has {} tiles)",
        row_tiles);
    TT_FATAL(in.dt_bias.logical_shape()[-1] == attrs.num_heads, "gdn_gates: dt_bias last dim must equal num_heads");
    TT_FATAL(in.a_neg.logical_shape()[-1] == attrs.num_heads, "gdn_gates: a_neg last dim must equal num_heads");
    TT_FATAL(
        in.dt_bias.padded_shape()[-1] == tt::constants::TILE_WIDTH &&
            in.a_neg.padded_shape()[-1] == tt::constants::TILE_WIDTH,
        "gdn_gates: dt_bias and a_neg must be a single tile wide");
}

GdnGatesOperation::spec_return_value_t GdnGatesOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const Shape shape =
        attrs.rank == 3 ? Shape({1, attrs.sequence, attrs.num_heads}) : Shape({1, 1, attrs.sequence, attrs.num_heads});
    TensorSpec spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config));
    return {spec, spec};
}

GdnGatesOperation::tensor_return_value_t GdnGatesOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {create_device_tensor(specs[0], in.gab.device()), create_device_tensor(specs[1], in.gab.device())};
}

std::tuple<Tensor, Tensor> gdn_gates(
    const Tensor& gab,
    const Tensor& dt_bias,
    const Tensor& a_neg,
    uint32_t a_col_offset,
    uint32_t b_col_offset,
    uint32_t num_heads,
    float beta_scale,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    TT_FATAL(
        a_col_offset % tt::constants::TILE_WIDTH == 0 && b_col_offset % tt::constants::TILE_WIDTH == 0,
        "gdn_gates: column offsets must be multiples of 32");
    const auto& s = gab.logical_shape();
    TT_FATAL(s.rank() == 3 || s.rank() == 4, "gdn_gates: gab must be [1,T,W] or [1,1,T,W]");
    // Same kernel numerics as the binary_ng chain: bf16 DEST (fp32 accumulation off) like the binary_ng chain, HiFi4,
    // exact SFPU.
    const auto kernel_config = init_device_compute_kernel_config(
        gab.device()->arch(),
        std::nullopt,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false,
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
    auto results = ttnn::device_operation::launch<GdnGatesOperation>(
        GdnGatesParams{
            .sequence = static_cast<uint32_t>(s[-2]),
            .rank = static_cast<uint32_t>(s.rank()),
            .num_heads = num_heads,
            .a_col_tile = a_col_offset / tt::constants::TILE_WIDTH,
            .b_col_tile = b_col_offset / tt::constants::TILE_WIDTH,
            .beta_scale = beta_scale,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = kernel_config},
        GdnGatesInputs{.gab = gab, .dt_bias = dt_bias, .a_neg = a_neg});
    return {results[0], results[1]};
}

}  // namespace ttnn::experimental::prim
