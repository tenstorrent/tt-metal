// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The entry point of ttnn.bringup.rms_norm: rms_norm_ttnn.py's validate() + rms_norm_ttnn(), in C++.
// The registry declarations (INPUT_TAGGERS / SUPPORTED / EXCLUSIONS) are the Python module's; this
// file enforces the same refusals in the same order and with the same exception types:
//   ValueErrorCpp        -> ValueError             (an input-contract violation)
//   UnsupportedAxisError -> UnsupportedAxisValue   (a SUPPORTED refusal, a NotImplementedError)
//   NotImplementedErrorCpp -> NotImplementedError  (the BAND scheme's output-placement limit)

#include "rms_norm_ttnn.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "device/rms_norm_ttnn_device_operation.hpp"
#include "device/rms_norm_ttnn_device_operation_types.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;
using ttnn::StorageType;

constexpr int64_t TILE_DIM = 32;

int64_t div_up(int64_t a, int64_t b) { return (a + b - 1) / b; }

std::vector<uint32_t> dims(const ttnn::Shape& s) {
    std::vector<uint32_t> v;
    v.reserve(s.rank());
    for (size_t i = 0; i < s.rank(); ++i) {
        v.push_back(s[i]);
    }
    return v;
}

std::string list_str(const std::vector<uint32_t>& v) { return fmt::format("[{}]", fmt::join(v, ", ")); }

bool on_device(const Tensor& t) { return t.storage_type() == StorageType::DEVICE; }

bool per_channel_dtype_ok(DataType dt) {
    return dt == DataType::FLOAT32 || dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B;
}

// rms_norm_ttnn_program_descriptor.py per_channel_form(): (is_blocked, channel_extent).
std::pair<bool, int64_t> per_channel_form(const Tensor& operand, int64_t width) {
    const auto shape = dims(operand.logical_shape());
    const int64_t wt = div_up(std::max<int64_t>(1, width), TILE_DIM);
    if (operand.layout() == Layout::ROW_MAJOR && shape.size() >= 2 && shape.back() == TILE_DIM) {
        int64_t folded = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) {
            folded *= shape[i];
        }
        if (folded == wt) {
            return {true, folded * TILE_DIM};
        }
    }
    return {false, shape.empty() ? 1 : shape.back()};
}

void check_per_channel(const char* name, const Tensor& operand, int64_t width) {
    if (!per_channel_dtype_ok(operand.dtype())) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: {} dtype {} is not in the accepted per-channel set [float32, bfloat16, bfloat8_b] (the "
            "same set at BOTH layouts; got layout {})",
            name,
            operand.dtype(),
            operand.layout()));
    }
    if (!on_device(operand)) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: {} must be resident on a device; got storage {}", name, operand.storage_type()));
    }
    const auto [blocked, channel_extent] = per_channel_form(operand, width);
    (void)blocked;
    if (channel_extent < width) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: {} covers {} channels, which is fewer than the input's last dimension {}; a per-channel "
            "operand must cover every channel (shape {}, layout {})",
            name,
            channel_extent,
            width,
            list_str(dims(operand.logical_shape())),
            operand.layout()));
    }
    if (operand.layout() == Layout::TILE) {
        const auto padded = dims(operand.padded_shape());
        const int64_t tile_padded_width = div_up(width, TILE_DIM) * TILE_DIM;
        const int64_t last = padded.empty() ? 1 : padded.back();
        if (last < tile_padded_width) {
            throw ValueErrorCpp(fmt::format(
                "rms_norm_ttnn: a TILE-layout {}'s padded last dim {} does not cover the input's tile-padded last dim "
                "{} (logical W = {})",
                name,
                last,
                tile_padded_width,
                width));
        }
        if (padded.size() >= 2 && padded[padded.size() - 2] != TILE_DIM) {
            throw ValueErrorCpp(fmt::format(
                "rms_norm_ttnn: a TILE-layout {}'s padded second-to-last dim must be one tile height ({}); got {} "
                "from shape {}",
                name,
                TILE_DIM,
                padded[padded.size() - 2],
                list_str(dims(operand.logical_shape()))));
        }
    }
}

void check_residual(const Tensor& residual, const Tensor& input) {
    if (!on_device(residual)) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: residual_input_tensor must be resident on a device; got storage {}",
            residual.storage_type()));
    }
    if (residual.dtype() != input.dtype()) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: residual_input_tensor dtype {} must equal the input's {}",
            residual.dtype(),
            input.dtype()));
    }
    if (residual.layout() != input.layout()) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: residual_input_tensor layout {} must equal the input's {}",
            residual.layout(),
            input.layout()));
    }
    if (dims(residual.logical_shape()) != dims(input.logical_shape())) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: residual_input_tensor shape {} must equal the input's {}",
            list_str(dims(residual.logical_shape())),
            list_str(dims(input.logical_shape()))));
    }
    if (dims(residual.padded_shape()) != dims(input.padded_shape())) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: residual_input_tensor padded shape {} must equal the input's {}",
            list_str(dims(residual.padded_shape())),
            list_str(dims(input.padded_shape()))));
    }
    const auto& rc = residual.memory_config();
    const auto& xc = input.memory_config();
    if (rc.memory_layout() != xc.memory_layout() || rc.shard_spec().has_value() != xc.shard_spec().has_value()) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: residual_input_tensor placement {} must equal the input's {} (shard spec included)",
            rc.memory_layout(),
            xc.memory_layout()));
    }
    if (rc.shard_spec().has_value() &&
        (rc.shard_spec()->shape != xc.shard_spec()->shape || rc.shard_spec()->grid != xc.shard_spec()->grid ||
         rc.shard_spec()->orientation != xc.shard_spec()->orientation)) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: residual_input_tensor shard spec {} must equal the input's {}",
            *rc.shard_spec(),
            *xc.shard_spec()));
    }
}

bool is_sharded_ml(TensorMemoryLayout ml) {
    return ml == TensorMemoryLayout::HEIGHT_SHARDED || ml == TensorMemoryLayout::WIDTH_SHARDED ||
           ml == TensorMemoryLayout::BLOCK_SHARDED;
}

bool supported_memory_layout(TensorMemoryLayout ml) {
    return ml == TensorMemoryLayout::INTERLEAVED || is_sharded_ml(ml);
}

// dest_helpers.hpp get_dest_limit(), host mirror.
uint32_t dest_tile_limit(const tt::tt_metal::ComputeConfigDescriptor& cfg) {
    if (cfg.dst_full_sync_en) {
        return cfg.fp32_dest_acc_en ? 8 : 16;
    }
    return cfg.fp32_dest_acc_en ? 4 : 8;
}

struct ResolvedProgramConfig {
    uint32_t subblock_w = 0;
    bool inplace = false;
};

ResolvedProgramConfig resolve_program_config(
    const Tensor& input, const std::optional<ProgramConfigArg>& program_config, uint32_t dest_limit) {
    const auto in_ml = input.memory_config().memory_layout();
    const bool sharded = is_sharded_ml(in_ml);
    if (!program_config.has_value()) {
        return {};
    }
    const auto& pc = *program_config;
    if (pc.use_welford) {
        throw ValueErrorCpp(
            "rms_norm_ttnn: program_config.use_welford=True is refused -- this op has no Welford single-pass "
            "statistics path; unset the field to get the two-pass reduce");
    }
    if (pc.sharded_variant != sharded) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: program_config variant does not match the input's placement -- a {} config was supplied "
            "for an {} input (memory_layout={})",
            pc.sharded_variant ? "SHARDED" : "DEFAULT",
            sharded ? "sharded" : "interleaved",
            in_ml));
    }
    if (!pc.sharded_variant) {
        return {};
    }
    const auto& ss = input.memory_config().shard_spec();
    TT_FATAL(ss.has_value(), "rms_norm_ttnn: a sharded memory_layout with no shard spec");
    const int64_t shard0 = ss->shape[0];
    const int64_t shard1 = ss->shape[1];
    const int64_t block_h = shard0 / TILE_DIM;
    const int64_t block_w = shard1 / TILE_DIM;

    if (pc.grid.has_value()) {
        const auto device_grid = input.device()->compute_with_storage_grid_size();
        const auto [gx, gy] = *pc.grid;
        if (!(1 <= gx && gx <= static_cast<int64_t>(device_grid.x) && 1 <= gy &&
              gy <= static_cast<int64_t>(device_grid.y))) {
            throw ValueErrorCpp(fmt::format(
                "rms_norm_ttnn: program_config.compute_with_storage_grid_size ({}, {}) is outside the device's "
                "compute grid ({}, {})",
                gx,
                gy,
                device_grid.x,
                device_grid.y));
        }
    }
    const std::pair<const char*, std::pair<int64_t, int64_t>> restated[] = {
        {"block_h", {pc.block_h.value_or(block_h), block_h}}, {"block_w", {pc.block_w.value_or(block_w), block_w}}};
    for (const auto& [name, v] : restated) {
        if (v.first != v.second) {
            throw ValueErrorCpp(fmt::format(
                "rms_norm_ttnn: program_config.{}={} disagrees with the input's shard geometry, which fixes {}={} "
                "(shard shape ({}, {}), tile {}). {} restates the shard spec and cannot express a different blocking",
                name,
                v.first,
                name,
                v.second,
                shard0,
                shard1,
                TILE_DIM,
                name));
        }
    }
    const int64_t subblock_w = pc.subblock_w;
    if (subblock_w < 1) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: program_config.subblock_w must be at least 1 (it is used as a divisor of the pass-B width "
            "block); got {}",
            subblock_w));
    }
    if (block_w % subblock_w != 0) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: program_config.subblock_w={} does not divide block_w={} (the shard's width in tiles), so "
            "an outer iteration would be partly empty",
            subblock_w,
            block_w));
    }
    if (subblock_w > dest_limit) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: program_config.subblock_w={} exceeds the DEST capacity {} tiles available at this "
            "compute config (fp32_dest_acc_en halves it); lower subblock_w or turn fp32_dest_acc_en off",
            subblock_w,
            dest_limit));
    }
    return {static_cast<uint32_t>(subblock_w), pc.inplace};
}

std::string alignment_tag(const std::vector<uint32_t>& shape) {
    const int64_t width = shape.size() >= 1 ? shape.back() : 1;
    const int64_t height = shape.size() >= 2 ? shape[shape.size() - 2] : TILE_DIM;
    if (width % TILE_DIM != 0) {
        return "w_non_aligned";
    }
    if (height % TILE_DIM != 0) {
        return "h_non_aligned";
    }
    return "tile_aligned";
}

// rms_norm_ttnn.py validate(): the runtime support gate, before any device work.
ResolvedProgramConfig validate(
    const Tensor& input,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ProgramConfigArg>& program_config,
    const tt::tt_metal::ComputeConfigDescriptor& cfg) {
    if (!on_device(input)) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: input_tensor must be resident on a device; got storage {}.  Move it with ttnn.to_device "
            "/ pass device= to ttnn.from_torch.",
            input.storage_type()));
    }
    const auto shape = dims(input.logical_shape());
    const int64_t width = shape.size() >= 1 ? shape.back() : 1;
    if (weight.has_value()) {
        check_per_channel("weight", *weight, width);
    }
    if (bias.has_value()) {
        check_per_channel("bias", *bias, width);
    }
    if (weight.has_value() && bias.has_value() && weight->layout() != bias->layout()) {
        throw ValueErrorCpp(fmt::format(
            "rms_norm_ttnn: weight and bias must share a layout when both are present; got weight layout {} and bias "
            "layout {}",
            weight->layout(),
            bias->layout()));
    }
    if (residual.has_value()) {
        check_residual(*residual, input);
    }

    // SUPPORTED, per axis, in the Python dict's order.
    const auto dt = input.dtype();
    if (!(dt == DataType::FLOAT32 || dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B)) {
        throw UnsupportedAxisError(fmt::format(
            "rms_norm_ttnn: dtype={} not in SUPPORTED [DataType.FLOAT32, DataType.BFLOAT16, DataType.BFLOAT8_B]", dt));
    }
    // fp32_dest_acc_en: both values supported.
    if (!(input.layout() == Layout::TILE || input.layout() == Layout::ROW_MAJOR)) {
        throw UnsupportedAxisError(
            fmt::format("rms_norm_ttnn: layout={} not in SUPPORTED [Layout.TILE, Layout.ROW_MAJOR]", input.layout()));
    }
    (void)alignment_tag(shape);  // every tag is SUPPORTED
    if (shape.size() > 5) {
        throw UnsupportedAxisError(
            fmt::format("rms_norm_ttnn: rank={} not in SUPPORTED [0, 1, 2, 3, 4, 5]", shape.size()));
    }
    // gamma_mode: all eight presence combinations are SUPPORTED; gamma_dtype was gated by check_per_channel.
    const Tensor* per_channel = weight.has_value() ? &*weight : (bias.has_value() ? &*bias : nullptr);
    if (per_channel != nullptr &&
        !(per_channel->layout() == Layout::TILE || per_channel->layout() == Layout::ROW_MAJOR)) {
        throw UnsupportedAxisError(fmt::format(
            "rms_norm_ttnn: gamma_layout={} not in SUPPORTED [Layout.TILE, Layout.ROW_MAJOR, 'none']",
            per_channel->layout()));
    }
    const auto in_ml = input.memory_config().memory_layout();
    if (!supported_memory_layout(in_ml)) {
        throw UnsupportedAxisError(fmt::format(
            "rms_norm_ttnn: memory_layout={} not in SUPPORTED [INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, "
            "BLOCK_SHARDED]",
            in_ml));
    }
    // EXCLUSIONS: empty.
    if (memory_config.has_value() && !supported_memory_layout(memory_config->memory_layout())) {
        throw UnsupportedAxisError(fmt::format(
            "rms_norm_ttnn: memory_config.memory_layout={} not in SUPPORTED [INTERLEAVED, HEIGHT_SHARDED, "
            "WIDTH_SHARDED, BLOCK_SHARDED]",
            memory_config->memory_layout()));
    }
    return resolve_program_config(input, program_config, dest_tile_limit(cfg));
}

bool same_placement_spec(const ttnn::MemoryConfig& a, const ttnn::MemoryConfig& b) {
    if (a.memory_layout() != b.memory_layout()) {
        return false;
    }
    if (!a.shard_spec().has_value() || !b.shard_spec().has_value()) {
        return !a.shard_spec().has_value() && !b.shard_spec().has_value();
    }
    return a.shard_spec()->shape == b.shard_spec()->shape && a.shard_spec()->grid == b.shard_spec()->grid &&
           a.shard_spec()->orientation == b.shard_spec()->orientation;
}

std::optional<Tensor> as_optional(const std::optional<const Tensor>& t) {
    if (!t.has_value()) {
        return std::nullopt;
    }
    return Tensor(*t);
}

}  // namespace

tt::tt_metal::ComputeConfigDescriptor default_compute_kernel_config() {
    tt::tt_metal::ComputeConfigDescriptor c;
    c.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    c.fp32_dest_acc_en = false;
    c.math_approx_mode = true;
    return c;
}

tt::tt_metal::ComputeConfigDescriptor normalize_compute_kernel_config(const std::optional<ComputeConfigArg>& cfg) {
    if (!cfg.has_value()) {
        return default_compute_kernel_config();
    }
    if (const auto* d = std::get_if<tt::tt_metal::ComputeConfigDescriptor>(&*cfg)) {
        return *d;
    }
    const auto& k = std::get<ttnn::DeviceComputeKernelConfig>(*cfg);
    auto out = default_compute_kernel_config();
    out.math_fidelity = k.math_fidelity;
    out.math_approx_mode = k.math_approx_mode;
    out.fp32_dest_acc_en = k.fp32_dest_acc_en;
    out.dst_full_sync_en = k.dst_full_sync_en;
    return out;
}

std::pair<ttnn::Tensor, bool> rms_norm_with_inplace(
    const ttnn::Tensor& input_tensor,
    double epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ProgramConfigArg>& program_config,
    const std::optional<ComputeConfigArg>& compute_kernel_config) {
    const auto cfg = normalize_compute_kernel_config(compute_kernel_config);
    const auto resolved =
        validate(input_tensor, weight, bias, residual_input_tensor, memory_config, program_config, cfg);
    const auto output_mem_config = memory_config.value_or(input_tensor.memory_config());

    // The ROW_MAJOR BAND scheme's output-placement limit (rms_norm_ttnn_program_descriptor.py _plan_band),
    // raised here so it keeps its type whatever the launch path does with exceptions.
    const auto in_ml = input_tensor.memory_config().memory_layout();
    if (input_tensor.layout() == Layout::ROW_MAJOR &&
        (in_ml == TensorMemoryLayout::WIDTH_SHARDED || in_ml == TensorMemoryLayout::BLOCK_SHARDED) &&
        input_tensor.logical_shape().volume() != 0) {
        const auto& out_mc = resolved.inplace ? input_tensor.memory_config() : output_mem_config;
        const bool band_out_local = same_placement_spec(input_tensor.memory_config(), out_mc);
        const auto out_ml = out_mc.memory_layout();
        if (!band_out_local && out_ml != TensorMemoryLayout::INTERLEAVED &&
            out_ml != TensorMemoryLayout::HEIGHT_SHARDED) {
            throw NotImplementedErrorCpp(fmt::format(
                "rms_norm_ttnn: a ROW_MAJOR {} input needs an output that is either the SAME shard spec (written in "
                "place) or stick-paged (INTERLEAVED / HEIGHT_SHARDED); got {} with a different geometry",
                in_ml,
                out_ml));
        }
    }

    auto out = ttnn::prim::bringup::rms_norm_ttnn(
        input_tensor,
        as_optional(weight),
        as_optional(bias),
        as_optional(residual_input_tensor),
        epsilon,
        cfg,
        resolved.subblock_w,
        resolved.inplace,
        output_mem_config);
    return {std::move(out), resolved.inplace};
}

ttnn::Tensor rms_norm(
    const ttnn::Tensor& input_tensor,
    double epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ProgramConfigArg>& program_config,
    const std::optional<ComputeConfigArg>& compute_kernel_config) {
    return rms_norm_with_inplace(
               input_tensor,
               epsilon,
               weight,
               bias,
               residual_input_tensor,
               memory_config,
               program_config,
               compute_kernel_config)
        .first;
}

}  // namespace ttnn::operations::bringup::rms_norm_ttnn
