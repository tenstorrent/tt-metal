// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.rms_norm's split program (CHANGELOG 6): device operation + program factory.  See
// rms_norm_split_device_operation.hpp for the options and kernels/rms_norm_split_compute.cpp for the math.

#include "rms_norm_split_device_operation.hpp"

#include <algorithm>
#include <bit>
#include <limits>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

constexpr uint32_t TILE = 32;
constexpr uint32_t MODE_LOCAL = 0;
constexpr uint32_t MODE_STATS_OUT = 1;
constexpr uint32_t MODE_STATS_IN = 2;

constexpr uint32_t CB_X = tt::CBIndex::c_0;
constexpr uint32_t CB_SCALER = tt::CBIndex::c_1;
constexpr uint32_t CB_ACC = tt::CBIndex::c_2;
constexpr uint32_t CB_STATS = tt::CBIndex::c_3;
constexpr uint32_t CB_RS = tt::CBIndex::c_4;
constexpr uint32_t CB_GAMMA = tt::CBIndex::c_5;
constexpr uint32_t CB_TMP = tt::CBIndex::c_6;
constexpr uint32_t CB_OUT = tt::CBIndex::c_7;

constexpr const char* KDIR = "ttnn/ttnn/bringup/rms_norm_ttnn/kernels/";

uint32_t mode_of(const RmsNormSplitParams& a, const RmsNormSplitInputs& t) {
    if (a.stats_out) {
        return MODE_STATS_OUT;
    }
    return t.partial_stats.has_value() ? MODE_STATS_IN : MODE_LOCAL;
}

uint32_t folded_rows(const Tensor& t) {
    const auto& p = t.padded_shape();
    return static_cast<uint32_t>(t.physical_volume() / p[-1]);
}

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// the op's grid: the device's compute grid, cut to max_grid (fork option, FIX.15)
CoreCoord op_grid(const RmsNormSplitParams& a, const Tensor& t) {
    auto g = t.device()->compute_with_storage_grid_size();
    if (a.max_grid.has_value()) {
        g.x = std::max<std::size_t>(1, std::min<std::size_t>(g.x, a.max_grid->x));
        g.y = std::max<std::size_t>(1, std::min<std::size_t>(g.y, a.max_grid->y));
    }
    return g;
}

uint32_t num_cores_of(const RmsNormSplitParams& a, const Tensor& t) {
    const auto g = op_grid(a, t);
    return static_cast<uint32_t>(g.x * g.y);
}

void check_tile_operand(const Tensor& t, const char* name) {
    TT_FATAL(t.storage_type() == ttnn::StorageType::DEVICE, "rms_norm (split): {} must be on a device", name);
    TT_FATAL(t.layout() == Layout::TILE, "rms_norm (split): {} must be TILE, got {}", name, t.layout());
    TT_FATAL(
        t.dtype() == DataType::FLOAT32 || t.dtype() == DataType::BFLOAT16,
        "rms_norm (split): {} must be float32 or bfloat16, got {}",
        name,
        t.dtype());
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "rms_norm (split): {} must be interleaved, got {}",
        name,
        t.memory_config().memory_layout());
}

}  // namespace

SplitPlan split_plan(const RmsNormSplitParams& a, const RmsNormSplitInputs& t, uint32_t num_cores) {
    SplitPlan p;
    p.Ht = folded_rows(t.input) / TILE;
    p.Wt_in = t.input.padded_shape()[-1] / TILE;
    p.c0t = a.col_start / TILE;
    p.Wt_win = (a.col_end - a.col_start) / TILE;
    p.G = t.partial_stats.has_value() ? t.partial_stats->padded_shape()[-1] / TILE : 0;
    const uint32_t mode = mode_of(a, t);
    const uint32_t nc = std::max<uint32_t>(1, num_cores);
    p.B = 1;
    if (mode != MODE_LOCAL) {
        // the fewest tile reads on the busiest core; the stats output keeps B <= 2 (its width is 32 B, and every
        // apply unit reads all of it after the gather)
        uint64_t best = std::numeric_limits<uint64_t>::max();
        const uint32_t max_b = mode == MODE_STATS_OUT ? std::min<uint32_t>(2, p.Wt_win) : p.Wt_win;
        for (uint32_t b = 1; b <= max_b; ++b) {
            if (p.Wt_win % b != 0) {
                continue;
            }
            const uint64_t per_unit = (p.Wt_win / b) * (mode == MODE_STATS_IN ? 2u : 1u) + p.G;
            const uint64_t cost = static_cast<uint64_t>(div_up(p.Ht * b, nc)) * per_unit;
            if (cost < best) {
                best = cost;
                p.B = b;
            }
        }
    }
    p.WB = p.Wt_win / p.B;
    uint32_t dest = a.fp32_dest_acc_en ? (a.dst_full_sync_en ? 8u : 4u) : (a.dst_full_sync_en ? 16u : 8u);
    if (t.weight.has_value() && mode != MODE_STATS_OUT) {
        dest /= 2;  // pass B holds the block's gamma tiles beside it in DEST
    }
    p.BLK = 1;
    for (uint32_t d = std::min(dest, p.WB); d >= 1; --d) {
        if (p.WB % d == 0) {
            p.BLK = d;
            break;
        }
    }
    return p;
}

void RmsNormSplitDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    check_tile_operand(t.input, "input_tensor");
    const uint32_t W = t.input.padded_shape()[-1];
    TT_FATAL(
        t.input.logical_shape()[-1] == W, "rms_norm (split): the input's last dim must be tile-aligned, got {}", W);
    TT_FATAL(
        a.col_start % TILE == 0 && a.col_end % TILE == 0 && a.col_start < a.col_end && a.col_end <= W,
        "rms_norm (split): input_window [{}, {}) must be tile-aligned inside [0, {}]",
        a.col_start,
        a.col_end,
        W);
    TT_FATAL(a.stats_width > 0, "rms_norm (split): stats_width must be positive");
    TT_FATAL(a.fp32_dest_acc_en, "rms_norm (split): needs fp32_dest_acc_en=True (the fp32 statistics)");
    TT_FATAL(
        !(a.stats_out && t.partial_stats.has_value()),
        "rms_norm (split): stats_out and partial_stats exclude each other");
    if (a.stats_out) {
        TT_FATAL(!t.weight.has_value(), "rms_norm (split): stats_out takes no weight");
    }
    if (t.partial_stats.has_value()) {
        const auto& s = *t.partial_stats;
        check_tile_operand(s, "partial_stats");
        TT_FATAL(
            folded_rows(s) == folded_rows(t.input),
            "rms_norm (split): partial_stats has {} rows, the input {}",
            folded_rows(s),
            folded_rows(t.input));
    }
    if (t.weight.has_value()) {
        const auto& w = *t.weight;
        check_tile_operand(w, "weight");
        const auto& wp = w.padded_shape();
        TT_FATAL(
            wp[-1] >= a.col_end - a.col_start,
            "rms_norm (split): weight covers {} channels, the window {}",
            wp[-1],
            a.col_end - a.col_start);
        TT_FATAL(folded_rows(w) == TILE, "rms_norm (split): weight must be one tile row ([1, 1, 1, W]), got {}", wp);
    }
}

RmsNormSplitDeviceOperation::spec_return_value_t RmsNormSplitDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& t) {
    auto shape = t.input.logical_shape();
    DataType dt = a.out_dtype;
    if (a.stats_out) {
        const auto plan = split_plan(a, t, num_cores_of(a, t.input));
        shape[-1] = TILE * plan.B;
        dt = DataType::FLOAT32;
    } else {
        shape[-1] = a.col_end - a.col_start;
    }
    return tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(dt, tt::tt_metal::PageConfig(Layout::TILE), a.output_mem_config));
}

RmsNormSplitDeviceOperation::tensor_return_value_t RmsNormSplitDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    if (t.output.has_value()) {  // output_tensor (P.16): the caller's buffer, exactly the op's output spec
        const auto& out = *t.output;
        const auto want = compute_output_specs(a, t);
        TT_FATAL(
            out.storage_type() == ttnn::StorageType::DEVICE && out.is_allocated(),
            "rms_norm_ttnn: output_tensor must be an allocated device tensor");
        TT_FATAL(
            out.tensor_spec() == want,
            "rms_norm_ttnn: output_tensor spec {} is not the op's output spec {}",
            out.tensor_spec(),
            want);
        TT_FATAL(
            out.buffer()->address() != t.input.buffer()->address(),
            "rms_norm_ttnn: output_tensor must not be the input (use program_config inplace)");
        return out;
    }
    return create_device_tensor(compute_output_specs(a, t), t.input.device());
}

ttsl::hash::hash_t RmsNormSplitDeviceOperation::compute_program_hash(
    const operation_attributes_t& a, const tensor_args_t& t) {
    auto spec_of = [](const std::optional<Tensor>& x) -> std::optional<tt::tt_metal::TensorSpec> {
        if (!x.has_value()) {
            return std::nullopt;
        }
        return x->tensor_spec();
    };
    return tt::tt_metal::operation::hash_operation<RmsNormSplitDeviceOperation>(
        std::bit_cast<uint32_t>(a.epsilon),
        a.stats_width,
        a.col_start,
        a.col_end,
        a.stats_out,
        a.out_dtype,
        a.output_mem_config,
        static_cast<uint32_t>(a.math_fidelity),
        a.fp32_dest_acc_en,
        a.math_approx_mode,
        a.dst_full_sync_en,
        t.input.tensor_spec(),
        spec_of(t.weight),
        spec_of(t.partial_stats),
        a.max_grid.has_value() ? a.max_grid->x : 0u,
        a.max_grid.has_value() ? a.max_grid->y : 0u);
}

tt::tt_metal::ProgramDescriptor RmsNormSplitProgramFactory::create_descriptor(
    const RmsNormSplitParams& a, const RmsNormSplitInputs& t, Tensor& output) {
    using namespace tt::tt_metal;
    const auto& input = t.input;
    const bool has_gamma = t.weight.has_value();
    const bool has_stats = t.partial_stats.has_value();
    const uint32_t mode = mode_of(a, t);

    const auto grid = op_grid(a, input);
    const CoreRange core_range({0, 0}, {grid.x - 1, grid.y - 1});
    const CoreRangeSet cores_set(core_range);
    const uint32_t num_cores = core_range.size();
    const auto plan = split_plan(a, t, num_cores);
    const uint32_t units = plan.Ht * plan.B;
    const uint32_t per_core = std::max<uint32_t>(1, div_up(units, num_cores));

    const auto fmt = [](DataType dt) { return datatype_to_dataformat_converter(dt); };
    const tt::DataFormat x_fmt = fmt(input.dtype());
    const tt::DataFormat out_fmt = fmt(output.dtype());
    const tt::DataFormat f32 = tt::DataFormat::Float32;
    const tt::DataFormat scaler_fmt = tt::DataFormat::Float16_b;  // 1.0, exact
    const tt::DataFormat stats_fmt = has_stats ? fmt(t.partial_stats->dtype()) : f32;
    const tt::DataFormat gamma_fmt = has_gamma ? fmt(t.weight->dtype()) : tt::DataFormat::Float16_b;

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t index, tt::DataFormat f, uint32_t pages) {
        const uint32_t page = tt::tile_size(f);
        desc.cbs.push_back(CBDescriptor{
            .total_size = pages * page,
            .core_ranges = cores_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index), .data_format = f, .page_size = page}}}});
    };
    add_cb(CB_X, x_fmt, 2 * plan.BLK);
    if (mode == MODE_STATS_OUT) {
        add_cb(CB_OUT, f32, 2);
    } else {
        add_cb(CB_SCALER, scaler_fmt, 1);
        add_cb(CB_RS, f32, 2);
        add_cb(CB_OUT, out_fmt, 2 * plan.BLK);
        if (mode == MODE_LOCAL) {
            add_cb(CB_ACC, f32, 1);
        } else {
            add_cb(CB_STATS, stats_fmt, 2 * plan.G);
        }
        if (has_gamma) {
            add_cb(CB_GAMMA, gamma_fmt, 2 * plan.BLK);
        }
    }

    std::vector<uint32_t> reader_ct = {
        CB_X,
        CB_SCALER,
        CB_STATS,
        CB_GAMMA,
        plan.Wt_in,
        plan.c0t,
        plan.WB,
        plan.BLK,
        plan.B,
        plan.G,
        mode,
        static_cast<uint32_t>(has_gamma)};
    TensorAccessorArgs(input.buffer()).append_to(reader_ct);
    TensorAccessorArgs(has_stats ? t.partial_stats->buffer() : nullptr).append_to(reader_ct);
    TensorAccessorArgs(has_gamma ? t.weight->buffer() : nullptr).append_to(reader_ct);

    const uint32_t wt_out = output.padded_shape()[-1] / TILE;
    std::vector<uint32_t> writer_ct = {CB_OUT, wt_out, plan.WB, plan.BLK, plan.B, mode};
    TensorAccessorArgs(output.buffer()).append_to(writer_ct);

    const float inv_w = 1.0f / static_cast<float>(a.stats_width);
    std::vector<uint32_t> compute_ct = {
        CB_X,
        CB_SCALER,
        CB_ACC,
        CB_STATS,
        CB_RS,
        CB_GAMMA,
        CB_TMP,
        CB_OUT,
        plan.WB,
        plan.BLK,
        plan.G,
        mode,
        static_cast<uint32_t>(has_gamma),
        std::bit_cast<uint32_t>(inv_w),
        std::bit_cast<uint32_t>(a.epsilon)};

    KernelDescriptor reader;
    reader.kernel_source = std::string(KDIR) + "rms_norm_split_reader.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores_set;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = std::string(KDIR) + "rms_norm_split_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores_set;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = std::string(KDIR) + "rms_norm_split_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores_set;
    compute.compile_time_args = std::move(compute_ct);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = a.math_fidelity,
        .fp32_dest_acc_en = a.fp32_dest_acc_en,
        .dst_full_sync_en = a.dst_full_sync_en,
        .math_approx_mode = a.math_approx_mode};

    Buffer* const stats_buf = has_stats ? t.partial_stats->buffer() : nullptr;
    Buffer* const gamma_buf = has_gamma ? t.weight->buffer() : nullptr;
    const auto cores = corerange_to_cores(core_range, num_cores, true);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord core = cores.at(i);
        const uint32_t us = std::min(i * per_core, units);
        const uint32_t ue = std::min(us + per_core, units);
        reader.emplace_runtime_args(core, {input.buffer(), stats_buf, gamma_buf, us, ue});
        writer.emplace_runtime_args(core, {output.buffer(), us, ue});
        compute.emplace_runtime_args(core, {ue - us});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::operations::bringup::rms_norm_ttnn

namespace ttnn::prim::bringup {

ttnn::Tensor rms_norm_split(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& weight,
    const std::optional<ttnn::Tensor>& partial_stats,
    const ttnn::operations::bringup::rms_norm_ttnn::RmsNormSplitParams& attrs,
    const std::optional<ttnn::Tensor>& output) {
    using Op = ttnn::operations::bringup::rms_norm_ttnn::RmsNormSplitDeviceOperation;
    auto args = Op::tensor_args_t{.input = input, .weight = weight, .partial_stats = partial_stats, .output = output};
    return ttnn::device_operation::launch<Op>(attrs, args);
}

}  // namespace ttnn::prim::bringup
