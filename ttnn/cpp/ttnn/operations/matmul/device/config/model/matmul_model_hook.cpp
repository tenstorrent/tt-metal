// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/model/matmul_model_hook.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <variant>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/config/model/matmul_model_selector.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt_stl/reflection.hpp"
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::matmul::model {

namespace {

// Headroom kept free below the L1 budget, for allocator alignment and small factory-side buffers (as the #58143
// selector this enumerator comes from)
constexpr std::uint32_t kL1HeadroomBytes = 16 * 1024;

// The model switches from the default config only when it predicts the default takes at least this many times as long
// as its pick: picks predicted within 2% of the default are within the model's error. When the default config can't be
// scored (MultiCore, or a layout the model doesn't cover), no margin applies.
constexpr double kMinPredictedGain = 1.02;

thread_local std::string last;

// ---- the model at matmul_model_path, loaded on the first eligible matmul and again when the path or file changes ----

// A model and the picks made with it (a pick holds only for the model that made it)
struct LoadedModel {
    std::filesystem::path path;
    std::unique_ptr<const ConfigModel> model;  // null when the blob at `path` can't be loaded
    // The file's modification time when a load failed (nullopt: there was no file), to try again once it changes
    std::optional<std::filesystem::file_time_type> failed_mtime;
    PickCache picks;
};

std::optional<std::filesystem::file_time_type> modification_time(const std::filesystem::path& path) {
    std::error_code error;
    const auto time = std::filesystem::last_write_time(path, error);
    return error ? std::nullopt : std::optional(time);
}

// The model at `path`: loaded once per path, and after a failed load again once the file appears or changes (a failure
// is logged once per state of the file, not once per call)
std::shared_ptr<LoadedModel> loaded_model(const std::filesystem::path& path) {
    static ttsl::Indestructible<std::mutex> mutex;
    static ttsl::Indestructible<std::shared_ptr<LoadedModel>> current;
    std::lock_guard lock(mutex.get());
    auto& loaded = current.get();
    if (loaded != nullptr && loaded->path == path &&
        (loaded->model != nullptr || loaded->failed_mtime == modification_time(path))) {
        return loaded;
    }
    auto next = std::make_shared<LoadedModel>();
    next->path = path;
    const auto mtime = modification_time(path);
    std::string error;
    next->model = ConfigModel::load_file(path.string(), error);
    if (next->model) {
        log_info(tt::LogOp, "matmul_model_config: loaded {} (model {})", path.string(), next->model->sha256());
    } else {
        next->failed_mtime = mtime;
        log_warning(
            tt::LogOp,
            "matmul_model_config is on but the matmul config model could not be loaded ({}); matmul keeps its default "
            "program configs until a model loads from matmul_model_path",
            error);
    }
    loaded = std::move(next);
    return loaded;
}

std::optional<DataType> model_dtype(tt::tt_metal::DataType t) {
    switch (t) {
        case tt::tt_metal::DataType::BFLOAT16: return DataType::kBfloat16;
        case tt::tt_metal::DataType::BFLOAT8_B: return DataType::kBfloat8B;
        case tt::tt_metal::DataType::BFLOAT4_B: return DataType::kBfloat4B;
        case tt::tt_metal::DataType::FLOAT32: return DataType::kFloat32;
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::UINT8:
        case tt::tt_metal::DataType::UINT16:
        case tt::tt_metal::DataType::INT32:
        case tt::tt_metal::DataType::FP8_E4M3:
        case tt::tt_metal::DataType::INT8:
        case tt::tt_metal::DataType::MXFP8_E4M3:
        case tt::tt_metal::DataType::MXFP8_E5M2:
        case tt::tt_metal::DataType::MXFP6_E2M3:
        case tt::tt_metal::DataType::MXFP6_E3M2:
        case tt::tt_metal::DataType::MXFP4:
        case tt::tt_metal::DataType::MXINT8:
        case tt::tt_metal::DataType::MXINT4:
        case tt::tt_metal::DataType::MXINT2:
        case tt::tt_metal::DataType::INVALID: return std::nullopt;
    }
    TT_THROW("Unreachable");
}

std::optional<Fidelity> model_fidelity(tt::tt_metal::MathFidelity f) {
    switch (f) {
        case tt::tt_metal::MathFidelity::LoFi: return Fidelity::kLoFi;
        case tt::tt_metal::MathFidelity::HiFi2: return Fidelity::kHiFi2;
        case tt::tt_metal::MathFidelity::HiFi3: return Fidelity::kHiFi3;
        case tt::tt_metal::MathFidelity::HiFi4: return Fidelity::kHiFi4;
        case tt::tt_metal::MathFidelity::Invalid: return std::nullopt;
    }
    TT_THROW("Unreachable");
}

bool is_32x32(const tt::tt_metal::Tile& tile) {
    return tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH;
}

std::uint32_t batch_of(const ttnn::Shape& shape) {
    return static_cast<std::uint32_t>(shape.volume() / (static_cast<std::uint64_t>(shape[-2]) * shape[-1]));
}

// Everything the model selector reads of a call, or the reason it keeps the default
struct Call {
    MatmulDesc desc;
    MatmulProblem problem;
    HardwareDesc hw;
    Arch arch = Arch::kBlackholeP150;
    std::string key;
};

std::optional<Call> describe(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attr,
    std::string& why) {
    auto* device = a.device();
    if (device == nullptr || b.device() == nullptr) {
        why = "inputs not on device";
        return std::nullopt;
    }
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        why = "not Blackhole";
        return std::nullopt;
    }
    if (attr.program_config.has_value() || attr.user_core_coord.has_value()) {
        why = "program_config or core_grid given";
        return std::nullopt;
    }
    if (attr.user_fused_activation.has_value() || attr.untilize_out || attr.global_cb.has_value() ||
        attr.sub_device_id.has_value() || !attr.prefetcher_pipes.empty() || attr.in0_column_offset.has_value()) {
        why = "fused activation, untilize_out, global CB, sub-device, prefetcher pipes or column offset";
        return std::nullopt;
    }
    if (!attr.compute_kernel_config.has_value() || !attr.output_dtype.has_value()) {
        why = "attributes not normalized";
        return std::nullopt;
    }
    if (a.layout() != ttnn::Layout::TILE || b.layout() != ttnn::Layout::TILE) {
        why = "not tile layout";
        return std::nullopt;
    }
    if (a.memory_config().is_sharded() || b.memory_config().is_sharded() || attr.output_mem_config.is_sharded()) {
        why = "sharded tensor";
        return std::nullopt;
    }
    const auto in0_tile = utilities::get_matmul_tile(a, attr.transpose_a);
    const auto in1_tile = utilities::get_matmul_tile(b, attr.transpose_b);
    if (!is_32x32(in0_tile) || !is_32x32(in1_tile) || (attr.output_tile.has_value() && !is_32x32(*attr.output_tile))) {
        why = "tiles other than 32x32";
        return std::nullopt;
    }
    const auto in0 = model_dtype(a.dtype());
    const auto in1 = model_dtype(b.dtype());
    const auto out = model_dtype(*attr.output_dtype);
    if (!in0 || !in1 || !out) {
        why = "dtype the model doesn't cover";
        return std::nullopt;
    }
    const auto a_padded = utilities::get_matmul_tensor_padded_shape(a, attr.transpose_a);
    const auto b_padded = utilities::get_matmul_tensor_padded_shape(b, attr.transpose_b);
    const auto a_logical = utilities::get_matmul_tensor_logical_shape(a, attr.transpose_a);
    const auto b_logical = utilities::get_matmul_tensor_logical_shape(b, attr.transpose_b);
    if (a_padded.rank() < 2 || b_padded.rank() < 2) {
        why = "rank below 2";
        return std::nullopt;
    }
    const auto [fidelity, approx, fp32_acc, l1_acc, full_sync] =
        get_compute_kernel_config_args(device->arch(), *attr.compute_kernel_config);
    const auto model_fid = model_fidelity(fidelity);
    if (!model_fid) {
        why = "invalid math fidelity";
        return std::nullopt;
    }

    Call call;
    MatmulDesc& p = call.desc;
    p.batch_a = batch_of(a_padded);
    p.batch_b = batch_of(b_padded);
    p.rank_a = a_padded.rank();
    p.rank_b = b_padded.rank();
    p.Mt = a_padded[-2] / tt::constants::TILE_HEIGHT;
    p.Kt = a_padded[-1] / tt::constants::TILE_WIDTH;
    p.Nt = b_padded[-1] / tt::constants::TILE_WIDTH;
    p.in0 = *in0;
    p.in1 = *in1;
    p.out = *out;
    if (bias.has_value()) {
        p.bias_tile_bytes =
            bias->tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(bias->dtype()));
        p.bias_rows = bias->padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        if (p.bias_rows != 1) {
            why = "bias other than a single row";
            return std::nullopt;
        }
    }
    p.transpose_a = attr.transpose_a;
    p.fidelity = *model_fid;
    p.fp32_dest_acc = fp32_acc;
    p.packer_l1_acc = l1_acc;
    p.dst_full_sync = full_sync;
    p.a = {false, a.memory_config().buffer_type() == tt::tt_metal::BufferType::L1};
    p.b = {false, b.memory_config().buffer_type() == tt::tt_metal::BufferType::L1};
    p.out_place = {false, attr.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1};
    if (auto reason = unsupported_reason(p); !reason.empty()) {
        why = reason;
        return std::nullopt;
    }
    if (p.batch_b > 1 && p.batch_a != p.batch_b) {
        // The model's problem has one batch count (A's): it would score a broadcast A as a single batch
        why = "B batched without matching A batches";
        return std::nullopt;
    }

    // The model's view: logical sizes in elements; it tells interleaved DRAM from interleaved L1
    MatmulProblem& m = call.problem;
    m.M = a_logical[-2];
    m.K = a_logical[-1];
    m.N = b_logical[-1];
    m.batch = batch_of(a_logical);
    m.a_dtype = *in0;
    m.b_dtype = *in1;
    m.has_out_dtype = true;
    m.out_dtype = *out;
    auto mem = [](bool in_l1) { return in_l1 ? MemLayout::kL1Interleaved : MemLayout::kDramInterleaved; };
    m.a_mem = mem(p.a.in_l1);
    m.b_mem = mem(p.b.in_l1);
    m.out_mem = mem(p.out_place.in_l1);
    m.fidelity = p.fidelity;
    m.fp32_dest_acc = fp32_acc;
    m.packer_l1_acc = l1_acc;
    m.bias = bias.has_value();
    m.activation = Activation::kNone;
    m.transpose_a = attr.transpose_a;
    m.transpose_b = attr.transpose_b;

    // Per-device grid and L1 (a mesh tensor's shapes are its per-device shapes)
    const auto grid = device->compute_with_storage_grid_size();
    const auto arch = blackhole_arch_for_grid(grid.x, grid.y);
    if (!arch) {
        why = fmt::format("no model device entry for a {}x{} grid", grid.x, grid.y);
        return std::nullopt;
    }
    call.arch = *arch;
    call.hw = HardwareDesc::blackhole(grid.x, grid.y, 0);
    call.hw.dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    // The pick cache key: an exact binary encoding of everything the pick depends on (the encoding the program
    // cache uses). Built on every call, so it must stay cheap: no text formatting.
    const bool has_bias = bias.has_value();
    call.key = ttsl::hash::canonical_key(
        a.logical_shape(),
        a.padded_shape(),
        b.logical_shape(),
        b.padded_shape(),
        a.dtype(),
        b.dtype(),
        *attr.output_dtype,
        a.memory_config(),
        b.memory_config(),
        attr.output_mem_config,
        attr.transpose_a,
        attr.transpose_b,
        has_bias ? std::optional(bias->padded_shape()) : std::nullopt,
        has_bias ? std::optional(bias->dtype()) : std::nullopt,
        has_bias ? std::optional(bias->memory_config()) : std::nullopt,
        fidelity,
        approx,
        fp32_acc,
        l1_acc,
        full_sync,
        attr.bcast_batch.value_or(false),
        device->arch(),
        grid.x,
        grid.y);
    return call;
}

// Live per-core L1 for this op's circular buffers: `max_l1_space` (free L1 above the lowest L1 buffer, the default
// selection's get_max_l1_space), less this op's interleaved L1 output and the headroom
std::uint32_t l1_cb_budget(const Tensor& a, const MatmulDesc& p, bool out_in_l1, std::uint32_t max_l1_space) {
    std::uint64_t budget = max_l1_space;
    if (out_in_l1) {
        const std::uint64_t out_tiles = static_cast<std::uint64_t>(std::max(p.batch_a, p.batch_b)) * p.Mt * p.Nt;
        const std::uint64_t banks = a.device()->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
        const std::uint64_t tile = tile_size_bytes(p.out);
        const std::uint64_t per_bank = (out_tiles + banks - 1) / banks * tile;
        budget = per_bank >= budget ? 0 : budget - per_bank;
    }
    return static_cast<std::uint32_t>(budget > kL1HeadroomBytes ? budget - kL1HeadroomBytes : 0);
}

// The program config matmul's default selection uses for this call (prim::matmul's get_program_config)
MatmulProgramConfig default_config(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attributes) {
    const std::uint32_t bias_tile_bytes =
        bias.has_value() ? tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(bias->dtype())) : 0;
    return ttnn::operations::matmul::get_program_config(
        a, b, attributes.transpose_a, attributes.transpose_b, bias_tile_bytes, attributes);
}

// K tiles per K block of the default config, if it has K blocks. The MultiCore config (the default's fallback)
// accumulates all of K in DST and packs each output tile once: one K block of Kt.
std::optional<std::uint32_t> default_in0_block_w(const MatmulProgramConfig& defaults, std::uint32_t Kt) {
    const auto in0_block_w = std::visit(
        [Kt](const auto& c) -> std::optional<std::uint32_t> {
            if constexpr (std::is_same_v<std::decay_t<decltype(c)>, MatmulMultiCoreProgramConfig>) {
                return Kt;
            } else if constexpr (requires { c.in0_block_w; }) {
                return static_cast<std::uint32_t>(c.in0_block_w);
            } else {
                return std::nullopt;
            }
        },
        defaults);
    return in0_block_w && *in0_block_w > 0 ? in0_block_w : std::nullopt;
}

std::uint32_t k_blocks(const MatmulDesc& p, std::uint32_t in0_block_w) {
    return (p.Kt + in0_block_w - 1) / in0_block_w;
}

// The most K blocks a pick may use, or nullopt when no bound applies (README, "Precision"; the model scores time, not
// accuracy). The K blocking is the part of a config that changes how partial sums are rounded, so a pick's result can
// differ slightly from the default's. One case is bounded: without packer L1 or fp32 accumulation the factories keep
// partial sums in the output format between K blocks, so with a block-float output every K block after the first
// re-quantizes them, and a pick uses at most the default config's K blocks (0, no pick, when the default has no K
// blocking to compare with). The default config depends on live L1 (get_max_l1_space), so a cached pick is checked
// again when that changes.
std::optional<std::uint32_t> max_k_blocks(const MatmulDesc& p, const MatmulProgramConfig& defaults) {
    if (p.packer_l1_acc || p.fp32_dest_acc || (p.out != DataType::kBfloat8B && p.out != DataType::kBfloat4B)) {
        return std::nullopt;
    }
    const auto in0_block_w = default_in0_block_w(defaults, p.Kt);
    return in0_block_w ? k_blocks(p, *in0_block_w) : 0;
}

// The default config as a candidate the model scores alongside the enumerated ones, when it is one of the modeled
// layouts (2D without transpose_mcast, 1D without gather_in0, Reuse) and has no fused activation: the model's pick has
// to beat it in the model's own prediction, or the default runs.
std::optional<Candidate> default_candidate(const MatmulProgramConfig& defaults, const MatmulDesc& p, HardwareDesc hw) {
    return std::visit(
        [&](const auto& c) -> std::optional<Candidate> {
            using C = std::decay_t<decltype(c)>;
            constexpr bool k2D = std::is_same_v<C, MatmulMultiCoreReuseMultiCastProgramConfig>;
            constexpr bool k1D = std::is_same_v<C, MatmulMultiCoreReuseMultiCast1DProgramConfig>;
            constexpr bool kReuse = std::is_same_v<C, MatmulMultiCoreReuseProgramConfig>;
            if constexpr (k2D || k1D || kReuse) {
                Blocking b;
                b.in0_block_w = static_cast<std::uint32_t>(c.in0_block_w);
                b.out_subblock_h = static_cast<std::uint32_t>(c.out_subblock_h);
                b.out_subblock_w = static_cast<std::uint32_t>(c.out_subblock_w);
                b.per_core_M = static_cast<std::uint32_t>(c.per_core_M);
                b.per_core_N = static_cast<std::uint32_t>(c.per_core_N);
                hw.grid_x = static_cast<std::uint32_t>(c.compute_with_storage_grid_size.x);
                hw.grid_y = static_cast<std::uint32_t>(c.compute_with_storage_grid_size.y);
                Family family = Family::kReuse;
                bool fuse_batch = true;
                if constexpr (k2D || k1D) {
                    if (c.fused_activation.has_value()) {
                        return std::nullopt;
                    }
                    b.out_block_h = static_cast<std::uint32_t>(c.out_block_h);
                    b.out_block_w = static_cast<std::uint32_t>(c.out_block_w);
                    fuse_batch = c.fuse_batch;
                }
                if constexpr (k2D) {
                    if (c.transpose_mcast) {
                        return std::nullopt;
                    }
                    family = Family::k2D;
                }
                if constexpr (k1D) {
                    if (c.gather_in0) {
                        return std::nullopt;
                    }
                    family = c.mcast_in0 ? Family::k1DMcastIn0 : Family::k1DMcastIn1;
                }
                return make_candidate(p, hw, family, b, fuse_batch);
            } else {
                return std::nullopt;
            }
        },
        defaults);
}

MatmulProgramConfig to_program_config(const Candidate& c) {
    const Blocking& b = c.blocking;
    const CoreCoord grid(c.grid_x, c.grid_y);
    switch (c.family) {
        case Family::k2D:
            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .out_block_h = b.out_block_h,
                .out_block_w = b.out_block_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .transpose_mcast = false,
                .fused_activation = std::nullopt,
                .fuse_batch = c.fuse_batch,
            };
        case Family::k1DMcastIn0:
        case Family::k1DMcastIn1:
            return MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .out_block_h = b.out_block_h,
                .out_block_w = b.out_block_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .fuse_batch = c.fuse_batch,
                .fused_activation = std::nullopt,
                .mcast_in0 = c.family == Family::k1DMcastIn0,
            };
        case Family::kReuse: break;
    }
    return MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = grid,
        .in0_block_w = b.in0_block_w,
        .out_subblock_h = b.out_subblock_h,
        .out_subblock_w = b.out_subblock_w,
        .per_core_M = b.per_core_M,
        .per_core_N = b.per_core_N,
    };
}

// The device op's own validation of `config` for this call (what would throw when the op is created): empty if it
// passes, else the error's first line
std::string validation_error(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const std::optional<Tensor>& output,
    const ttnn::prim::MatmulParams& attributes,
    MatmulProgramConfig config) {
    try {
        normalize_program_config(config, a.device()->compute_with_storage_grid_size());
        ttnn::prim::MatmulParams with_config = attributes;
        with_config.program_config = config;
        const ttnn::prim::MatmulInputs args{{a, b}, {bias}, {output}};
        ttnn::prim::MatmulDeviceOperation::validate_on_program_cache_miss(with_config, args);
        ttnn::prim::MatmulDeviceOperation::compute_output_specs(with_config, args);
        return "";
    } catch (const std::exception& e) {
        const std::string what = e.what();
        // tt_throw's what(): "<kind> @ file:line: condition\ninfo:\n<message>\nbacktrace:..."
        constexpr std::string_view kInfoMarker = "info:\n";
        const auto info = what.find(kInfoMarker);
        const std::string msg = info == std::string::npos ? what : what.substr(info + kInfoMarker.size());
        return msg.substr(0, msg.find('\n'));
    } catch (...) {
        return "unknown error";
    }
}

std::optional<MatmulProgramConfig> choose(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const std::optional<Tensor>& output,
    const ttnn::prim::MatmulParams& attributes) {
    const auto start = std::chrono::steady_clock::now();
    std::string why;
    auto call = describe(a, b, bias, attributes, why);
    if (!call) {
        last = "default: " + why;
        return std::nullopt;
    }
    // Held for the whole call, so a reload for another thread's call doesn't unload the model in use. Like the rest of
    // ttnn.CONFIG, matmul_model_path is unsynchronized: it must not be set while another thread runs a matmul.
    const auto loaded = loaded_model(ttnn::CONFIG.get<"matmul_model_path">());
    const ConfigModel* m = loaded->model.get();
    if (m == nullptr) {
        last = "default: model not loaded";
        return std::nullopt;
    }
    // Picks are kept per model, so the call signature alone identifies a pick
    PickCache& picks = loaded->picks;
    const std::string& key = call->key;
    // A preallocated output is already outside the free L1 the budget starts from
    const std::uint32_t max_l1_space = utilities::get_max_l1_space(a);
    const std::uint32_t budget =
        l1_cb_budget(a, call->desc, call->desc.out_place.in_l1 && !output.has_value(), max_l1_space);
    call->hw.l1_cb_budget = budget;

    auto use = [&](const Pick& pick) -> std::optional<MatmulProgramConfig> {
        if (!pick.candidate) {
            last = "default: " + pick.description;
            return std::nullopt;
        }
        if (pick.cb_bytes > budget) {
            last = fmt::format("default: cached pick needs {} B of L1, {} B free now", pick.cb_bytes, budget);
            return std::nullopt;
        }
        last = "model: " + pick.description;
        return to_program_config(*pick.candidate);
    };
    const auto cached = picks.find(key);
    // A pick that failed for lack of L1 is made again once more L1 is free
    if (cached && !(cached->failed_budget && budget > *cached->failed_budget)) {
        // The pick kept to the precision bound in the L1 it was made in; with other L1 the default's K blocking may
        // differ
        if (cached->candidate && cached->max_l1_space != max_l1_space) {
            const auto limit = max_k_blocks(call->desc, default_config(a, b, bias, attributes));
            const std::uint32_t blocks = k_blocks(call->desc, cached->candidate->blocking.in0_block_w);
            if (limit && blocks > *limit) {
                last = fmt::format(
                    "default: cached pick's {} K blocks are more than the default config's {} now", blocks, *limit);
                return std::nullopt;
            }
        }
        return use(*cached);
    }

    auto candidates = enumerate_candidates_with_narrower_grid(call->desc, call->hw);
    std::string none_left = "no candidate fits L1";
    if (bias.has_value()) {
        // The reuse factory doesn't fuse a row bias (matmul post-processes it for an explicit Reuse config)
        std::erase_if(candidates, [](const Candidate& c) { return c.family == Family::kReuse; });
    }
    // Keep only the candidates within the precision bound (max_k_blocks), and score the default config with them
    // (default_candidate)
    std::optional<std::size_t> default_index;
    if (!candidates.empty()) {
        const MatmulProgramConfig defaults = default_config(a, b, bias, attributes);
        if (const auto limit = max_k_blocks(call->desc, defaults)) {
            std::erase_if(
                candidates, [&](const Candidate& c) { return k_blocks(call->desc, c.blocking.in0_block_w) > *limit; });
            none_left = *limit > 0
                            ? fmt::format("no candidate fits L1 with at most the default config's {} K blocks", *limit)
                            : "block-float output and the default config has no K blocking to compare with";
        }
        const auto own = default_candidate(defaults, call->desc, call->hw);
        if (own && !candidates.empty()) {
            default_index = candidates.size();
            candidates.push_back(*own);
        }
    }
    const std::vector<double> predicted = m->predict_ns(call->problem, candidates, call->arch);
    Pick pick;
    pick.description = candidates.empty() ? none_left : "the model can't score any candidate";
    std::size_t rejected_for_l1 = 0;
    bool rejected_by_validation = false;
    bool default_predicted_fastest = false;
    for (std::size_t i : rank_by_prediction(predicted)) {
        if (default_index && i != *default_index && predicted[*default_index] < kMinPredictedGain * predicted[i]) {
            pick.description = fmt::format(
                "the model predicts {:.0f} ns for its best candidate and {:.0f} ns for the default config, less than "
                "{:.0f}% apart",
                predicted[i],
                predicted[*default_index],
                100 * (kMinPredictedGain - 1));
            default_predicted_fastest = true;
            break;
        }
        if (i == default_index) {
            pick.description = fmt::format(
                "the model predicts the default config fastest ({:.0f} ns, rank {} of {} candidates)",
                predicted[i],
                rejected_for_l1 + 1,
                candidates.size());
            default_predicted_fastest = true;
            break;
        }
        const Candidate& c = candidates[i];
        const std::uint32_t bytes = circular_buffer_bytes(call->desc, call->hw, c);
        if (bytes > budget) {
            ++rejected_for_l1;
            continue;
        }
        // The enumerator emits only configs the device op accepts (MatmulModelCandidates.AllPassTheDeviceOpRules), so
        // this check is a backstop and a rejection is a selector bug: the default runs rather than the next candidate.
        // A rejection is a TT_FATAL (a critical log line, a backtrace, abort() under TT_ASSERT_ABORT), so a pick makes
        // at most one.
        const MatmulProgramConfig config = to_program_config(c);
        if (auto error = validation_error(a, b, bias, output, attributes, config); !error.empty()) {
            pick.description = fmt::format("the device op rejects the model's pick {}: {}", config, error);
            log_warning(
                tt::LogOp, "matmul_model_config: {}; matmul keeps its default program config", pick.description);
            rejected_by_validation = true;
            break;
        }
        pick.candidate = c;
        pick.cb_bytes = bytes;
        pick.max_l1_space = max_l1_space;
        pick.description = fmt::format(
            "{} (predicted {:.0f} ns, rank {} of {} candidates{})",
            config,
            predicted[i],
            rejected_for_l1 + 1,
            candidates.size(),
            default_index ? fmt::format("; default config predicted {:.0f} ns", predicted[*default_index]) : "");
        break;
    }
    if (!pick.candidate && !rejected_by_validation && !default_predicted_fastest && rejected_for_l1 > 0) {
        pick.description = fmt::format("all {} scored candidates need more L1 than is free", rejected_for_l1);
    }
    // Retried once more L1 is free: with more L1 there are more candidates, including ones within the precision bound
    if (!pick.candidate && !rejected_by_validation && !default_predicted_fastest &&
        (candidates.empty() || rejected_for_l1 > 0)) {
        pick.failed_budget = budget;
    }
    // Time of this selection; later calls reuse the pick and its description
    pick.description += fmt::format(
        " (selected in {} us)",
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
    return use(cached ? picks.replace(key, std::move(pick)) : picks.insert(key, pick));
}

}  // namespace

std::optional<MatmulProgramConfig> model_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const std::optional<Tensor>& optional_output_tensor,
    const ttnn::prim::MatmulParams& attributes) {
    if (!ttnn::CONFIG.get<"matmul_model_config">()) {
        return std::nullopt;
    }
    try {
        return choose(input_tensor_a, input_tensor_b, bias, optional_output_tensor, attributes);
    } catch (const std::exception& e) {
        last = std::string("default: selector error: ") + e.what();
    } catch (...) {
        last = "default: selector error";
    }
    log_warning(tt::LogOp, "matmul_model_config: {}; matmul keeps its default program config", last);
    return std::nullopt;
}

std::string last_pick() { return last; }

}  // namespace ttnn::operations::matmul::model
