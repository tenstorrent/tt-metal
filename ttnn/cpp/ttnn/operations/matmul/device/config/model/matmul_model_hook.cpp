// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/model/matmul_model_hook.hpp"

#include <exception>
#include <memory>
#include <mutex>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/model/matmul_model_selector.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::matmul::model {

namespace {

// Headroom kept free below the L1 budget, for allocator alignment and small factory-side buffers (as the #57884
// selector this enumerator comes from)
constexpr std::uint32_t kL1HeadroomBytes = 16 * 1024;

thread_local std::string last;

// ---- the model, loaded once on the first eligible matmul ----

struct LoadedModel {
    std::once_flag once;
    std::unique_ptr<const ConfigModel> model;
};

LoadedModel& loaded_model() {
    static LoadedModel loaded;
    return loaded;
}

const ConfigModel* model() {
    auto& loaded = loaded_model();
    std::call_once(loaded.once, [&] {
        const std::string path = ttnn::CONFIG.get<"matmul_model_path">();
        std::string error;
        loaded.model = ConfigModel::load_file(path, error);
        if (loaded.model) {
            log_info(tt::LogOp, "matmul_model_config: loaded {} (model {})", path, loaded.model->sha256());
        } else {
            log_warning(
                tt::LogOp,
                "matmul_model_config is on but the matmul config model could not be loaded ({}); matmul keeps its "
                "default program configs",
                error);
        }
    });
    return loaded.model.get();
}

PickCache& pick_cache() {
    static PickCache cache;
    return cache;
}

std::optional<DataType> model_dtype(tt::tt_metal::DataType t) {
    switch (t) {
        case tt::tt_metal::DataType::BFLOAT16: return DataType::kBfloat16;
        case tt::tt_metal::DataType::BFLOAT8_B: return DataType::kBfloat8B;
        case tt::tt_metal::DataType::BFLOAT4_B: return DataType::kBfloat4B;
        case tt::tt_metal::DataType::FLOAT32: return DataType::kFloat32;
        default: return std::nullopt;
    }
}

Fidelity model_fidelity(tt::tt_metal::MathFidelity f) {
    switch (f) {
        case tt::tt_metal::MathFidelity::LoFi: return Fidelity::kLoFi;
        case tt::tt_metal::MathFidelity::HiFi2: return Fidelity::kHiFi2;
        case tt::tt_metal::MathFidelity::HiFi3: return Fidelity::kHiFi3;
        default: return Fidelity::kHiFi4;
    }
}

bool is_32x32(const tt::tt_metal::Tile& tile) { return tile.get_height() == 32 && tile.get_width() == 32; }

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

    Call call;
    MatmulDesc& p = call.desc;
    p.batch_a = batch_of(a_padded);
    p.batch_b = batch_of(b_padded);
    p.rank_a = a_padded.rank();
    p.rank_b = b_padded.rank();
    p.Mt = a_padded[-2] / 32;
    p.Kt = a_padded[-1] / 32;
    p.Nt = b_padded[-1] / 32;
    p.in0 = *in0;
    p.in1 = *in1;
    p.out = *out;
    if (bias.has_value()) {
        p.bias_tile_bytes =
            bias->tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(bias->dtype()));
        p.bias_rows = bias->padded_shape()[-2] / 32;
        if (p.bias_rows != 1) {
            why = "bias other than a single row";
            return std::nullopt;
        }
    }
    p.transpose_a = attr.transpose_a;
    p.fidelity = model_fidelity(fidelity);
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

    call.key = fmt::format(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}x{}",
        a.logical_shape(),
        a.padded_shape(),
        b.logical_shape(),
        b.padded_shape(),
        static_cast<int>(a.dtype()),
        static_cast<int>(b.dtype()),
        static_cast<int>(*attr.output_dtype),
        a.memory_config(),
        b.memory_config(),
        attr.output_mem_config,
        attr.transpose_a,
        attr.transpose_b,
        bias.has_value()
            ? fmt::format("{}/{}/{}", bias->padded_shape(), static_cast<int>(bias->dtype()), bias->memory_config())
            : std::string("none"),
        static_cast<int>(fidelity),
        approx,
        fmt::format("{}{}{}", fp32_acc, l1_acc, full_sync),
        attr.bcast_batch.value_or(false),
        static_cast<int>(device->arch()),
        grid.x,
        grid.y);
    return call;
}

// Live per-core L1 for this op's circular buffers: free L1 above the lowest L1 buffer (the default selection's
// get_max_l1_space), less this op's interleaved L1 output and the headroom
std::uint32_t l1_cb_budget(const Tensor& a, const MatmulDesc& p, bool out_in_l1) {
    std::uint64_t budget = utilities::get_max_l1_space(a);
    if (out_in_l1) {
        const std::uint64_t out_tiles = static_cast<std::uint64_t>(std::max(p.batch_a, p.batch_b)) * p.Mt * p.Nt;
        const std::uint64_t banks = a.device()->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
        const std::uint64_t tile = p.out == DataType::kFloat32    ? 4096
                                   : p.out == DataType::kBfloat16 ? 2048
                                   : p.out == DataType::kBfloat8B ? 1088
                                                                  : 576;
        const std::uint64_t per_bank = (out_tiles + banks - 1) / banks * tile;
        budget = per_bank >= budget ? 0 : budget - per_bank;
    }
    return static_cast<std::uint32_t>(budget > kL1HeadroomBytes ? budget - kL1HeadroomBytes : 0);
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
        const auto info = what.find("info:\n");
        const std::string msg = info == std::string::npos ? what : what.substr(info + 6);
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
    std::string why;
    auto call = describe(a, b, bias, attributes, why);
    if (!call) {
        last = "default: " + why;
        return std::nullopt;
    }
    const ConfigModel* m = model();
    if (m == nullptr) {
        last = "default: model not loaded";
        return std::nullopt;
    }
    const std::string key = call->key + "|" + m->sha256();
    const std::uint32_t budget = l1_cb_budget(a, call->desc, call->desc.out_place.in_l1);
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
    if (auto cached = pick_cache().find(key)) {
        return use(*cached);
    }

    auto candidates = enumerate_candidates(call->desc, call->hw);
    if (bias.has_value()) {
        // The reuse factory doesn't fuse a row bias (matmul post-processes it for an explicit Reuse config)
        std::erase_if(candidates, [](const Candidate& c) { return c.family == Family::kReuse; });
    }
    const std::vector<double> predicted = m->predict_ns(call->problem, candidates, call->arch);
    Pick pick;
    pick.description = candidates.empty() ? "no candidate fits L1" : "the model can't score any candidate";
    std::size_t rejected = 0;
    for (std::size_t i : rank_by_prediction(predicted)) {
        const Candidate& c = candidates[i];
        const std::uint32_t bytes = circular_buffer_bytes(call->desc, call->hw, c);
        if (bytes > budget) {
            ++rejected;
            continue;
        }
        const MatmulProgramConfig config = to_program_config(c);
        if (auto error = validation_error(a, b, bias, output, attributes, config); !error.empty()) {
            log_debug(tt::LogOp, "matmul_model_config: candidate {} rejected by validation: {}", config, error);
            ++rejected;
            continue;
        }
        pick.candidate = c;
        pick.cb_bytes = bytes;
        pick.description = fmt::format(
            "{} (predicted {:.0f} ns, rank {} of {} candidates)",
            config,
            predicted[i],
            rejected + 1,
            candidates.size());
        break;
    }
    if (!pick.candidate && rejected > 0) {
        pick.description = fmt::format("all {} scored candidates rejected", rejected);
    }
    return use(pick_cache().insert(key, std::move(pick)));
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
    return std::nullopt;
}

std::string last_pick() { return last; }

}  // namespace ttnn::operations::matmul::model
