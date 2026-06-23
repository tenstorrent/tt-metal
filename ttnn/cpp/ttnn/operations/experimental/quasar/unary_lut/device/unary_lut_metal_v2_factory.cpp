// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) program factory for the unary_lut op.
//
// The UNARY analog of binary_ng_metal_v2_factory.cpp: ONE input DFB (in0) + an
// output DFB (out) instead of two inputs, and a piecewise-LUT SFPU compute kernel
// instead of ADD. The fully-sharded fast path: each input/output shard is borrowed
// by a DFB (DataflowBufferSpec::borrowed_from -> the tensor's TensorParameter), the
// reader/writer do no NoC work for resident shards, and the compute kernel runs in
// local L1.

#include "unary_lut_device_operation.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::operations::experimental::quasar::unary_lut {

namespace {

using ttnn::device_operation::ProgramArtifacts;

constexpr const char* kReaderDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary_lut/device/kernels_dfb/dataflow/reader_sharded_dfb.cpp";
constexpr const char* kWriterDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary_lut/device/kernels_dfb/dataflow/writer_sharded_dfb.cpp";
constexpr const char* kComputeDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary_lut/device/kernels_dfb/compute/unary_lut_dfb.cpp";

// Format a float as a C++ source literal that round-trips exactly (hex-free, max
// precision). Uses std::scientific with max_digits10 so the mantissa ALWAYS contains a
// '.' and the value carries an exponent — this guarantees the 'f' suffix attaches to a
// floating literal, never to an integer (e.g. "10f" / "0f" are ill-formed and trigger
// the compiler's user-defined-literal lookup `operator""f`).
std::string float_literal(float v) {
    std::ostringstream os;
    os.setf(std::ios::scientific, std::ios::floatfield);
    os.precision(8);  // 9 significant digits (max_digits10 for fp32) -> exact round-trip
    os << v << "f";
    return os.str();
}

// Build the kernel -D defines that bake `lut` into the compute kernel (unary_lut_sfpu.h).
// Mirrors the tt-llk generic-LUT build header: LUT_EVAL_METHOD / LUT_*_DEGREE /
// LUT_NUM_SEGMENTS / LUT_DATA_INIT. When `lut` is nullopt the kernel keeps its
// compile-time default (deg-2 / 4-seg sigmoid).
m2::KernelSpec::CompilerOptions::Defines make_lut_defines(const std::optional<LutConfig>& lut) {
    m2::KernelSpec::CompilerOptions::Defines defines;
    if (!lut.has_value()) {
        return defines;
    }
    const auto& l = *lut;
    defines.emplace("LUT_EVAL_METHOD", std::to_string(l.eval_method));
    defines.emplace("LUT_POLY_DEGREE", std::to_string(l.poly_degree));
    defines.emplace("LUT_NUM_SEGMENTS", std::to_string(l.num_segments));
    defines.emplace("LUT_NUM_DEGREE", std::to_string(l.num_degree));
    defines.emplace("LUT_DEN_DEGREE", std::to_string(l.den_degree));

    std::ostringstream os;
    os << "{";
    for (std::size_t i = 0; i < l.data.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << float_literal(l.data[i]);
    }
    os << "}";
    defines.emplace("LUT_DATA_INIT", os.str());

    // ---- Range-reduction defines. Mirror the tt-llk generic-LUT build header: the
    // kernel does reduce-then-poly-then-reconstruct guarded by LUT_RR_METHOD. When
    // rr_method == 0 (none) emit nothing so the kernel keeps its no-RR default and the
    // build is byte-identical to the no-RR path. Method-specific params are emitted for
    // ALL methods (the kernel's #ifndef fallbacks make extras harmless), driven solely
    // by the LutConfig the driver parsed from the CSV — no per-activation special-casing.
    if (l.rr_method != 0) {
        defines.emplace("LUT_RR_METHOD", std::to_string(l.rr_method));
        defines.emplace("LUT_RR_LOG_LN2", float_literal(l.rr_log_ln2));
        defines.emplace("LUT_RR_EXP_MULT", float_literal(l.rr_exp_mult));
        defines.emplace("LUT_RR_EXP_CONST", float_literal(l.rr_exp_const));
        defines.emplace("LUT_RR_SCALE0", float_literal(l.rr_scale0));
        defines.emplace("LUT_RR_SCALE1", float_literal(l.rr_scale1));
        defines.emplace("LUT_RR_SCALE2", float_literal(l.rr_scale2));
        defines.emplace("LUT_RR_EXP2_MULT", float_literal(l.rr_exp2_mult));
        defines.emplace("LUT_RR_COMPOSE", std::to_string(l.rr_compose));
        defines.emplace("LUT_RR_LOG2_SCALE", float_literal(l.rr_log2_scale));
        defines.emplace("LUT_RR_LOG2_BASIS_MMINUS1", std::to_string(l.rr_log2_basis_mminus1));
        defines.emplace("LUT_RR_INPUT_OFFSET", float_literal(l.rr_input_offset));
        defines.emplace("LUT_RR_POW_N", std::to_string(l.rr_pow_n));
        defines.emplace("LUT_RR_POW_RECIP", std::to_string(l.rr_pow_recip));

        // Newton-root (method 9) standalone evaluator constants. The magic seed is a
        // 32-bit pattern reinterpreted as an int in the kernel, so emit it as a hex
        // literal (round-trips exactly). C1/C2 are float literals; iters/n/reciprocal
        // are integers (the kernel selects the sqrt vs rsqrt vs cbrt body by VALUE on
        // LUT_NR_N / LUT_NR_RECIPROCAL).
        if (l.rr_method == 9) {
            std::ostringstream magic_os;
            magic_os << "0x" << std::hex << l.nr_magic << "u";
            defines.emplace("LUT_NR_MAGIC", magic_os.str());
            defines.emplace("LUT_NR_C1", float_literal(l.nr_c1));
            defines.emplace("LUT_NR_C2", float_literal(l.nr_c2));
            defines.emplace("LUT_NR_ITERS", std::to_string(l.nr_iters));
            defines.emplace("LUT_NR_N", std::to_string(l.nr_n));
            defines.emplace("LUT_NR_RECIPROCAL", std::to_string(l.nr_reciprocal));
        }
    }

    // ---- Asymptotic-factoring defines. For a deployed pick whose tail segments are fit as
    // f(x) = dominant(x) * correction(x), the driver parsed the per-segment is_asymptotic
    // column into asym_mask (bit SEG => segment SEG asymptotic) and the shared dominant_factor
    // class into dom_class. Emit them so the kernel multiplies the per-segment Horner result by
    // dominant(x). dom_class == 0 => emit nothing (kernel default = no factoring, byte-identical
    // to the bare-poly cascade). Generic over all classes; no per-activation special-casing.
    if (l.dom_class != 0) {
        defines.emplace("LUT_ASYM_MASK", std::to_string(l.asym_mask) + "u");
        defines.emplace("LUT_DOMINANT_CLASS", std::to_string(l.dom_class));
    }
    return defines;
}

// Per-core shard tile count (tile-rounded shard height-in-tiles * width-in-tiles).
uint32_t shard_tiles(const Tensor& tensor, const ShardSpec& shard_spec) {
    const uint32_t tile_h = tensor.tensor_spec().tile().get_height();
    const uint32_t tile_w = tensor.tensor_spec().tile().get_width();
    const uint32_t shard_ht = tt::round_up(shard_spec.shape[0], tile_h) / tile_h;
    const uint32_t shard_wt = tt::round_up(shard_spec.shape[1], tile_w) / tile_w;
    return shard_ht * shard_wt;
}

}  // namespace

ProgramArtifacts UnaryLutDeviceOperation::ProgramFactoryMetalV2::create_program_artifacts(
    const operation_attributes_t& op, const tensor_args_t& tensor_args, Tensor& out) {
    const Tensor& in = tensor_args.input_tensor;

    TT_FATAL(in.is_sharded() && out.is_sharded(), "unary_lut DFB factory requires sharded input and output");

    Buffer* in_buffer = in.buffer();
    Buffer* out_buffer = out.buffer();
    TT_FATAL(in_buffer != nullptr && out_buffer != nullptr, "unary_lut DFB factory requires allocated device buffers");

    const tt::DataFormat in_df = datatype_to_dataformat_converter(in.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(out.dtype());
    const tt::tt_metal::Tile in_tile = in.tensor_spec().tile();
    const tt::tt_metal::Tile out_tile = out.tensor_spec().tile();
    const uint32_t in_tile_bytes = static_cast<uint32_t>(in_tile.get_tile_size(in_df));
    const uint32_t out_tile_bytes = static_cast<uint32_t>(out_tile.get_tile_size(out_df));

    const ShardSpec in_shard = *in.shard_spec();
    const ShardSpec out_shard = *out.shard_spec();
    const CoreRangeSet& grid = op.worker_grid;
    const auto cores = corerange_to_cores(grid, std::nullopt, /*row_wise=*/true);
    TT_FATAL(!cores.empty(), "unary_lut DFB factory: empty worker grid");

    const uint32_t in_shard_tiles = shard_tiles(in, in_shard);
    const uint32_t out_shard_tiles = shard_tiles(out, out_shard);
    TT_FATAL(
        in_shard_tiles == out_shard_tiles,
        "unary_lut DFB factory: input/output shard tile counts must match but got {} {}",
        in_shard_tiles,
        out_shard_tiles);
    const uint32_t n_shard_tiles = in_shard_tiles;

    // Tiles processed per DST register acquire (bounded by DST capacity; 8 for 16-bit).
    const uint32_t num_tiles_per_cycle = std::min<uint32_t>(8u, n_shard_tiles);

    const m2::TensorParamName T_IN{"unary_lut_in"};
    const m2::TensorParamName T_OUT{"unary_lut_out"};
    const m2::DFBSpecName IN0{"unary_lut_in0_dfb"};
    const m2::DFBSpecName OUT{"unary_lut_out_dfb"};
    const m2::KernelSpecName READER{"unary_lut_reader"};
    const m2::KernelSpecName WRITER{"unary_lut_writer"};
    const m2::KernelSpecName COMPUTE{"unary_lut_compute"};

    auto make_dfb = [](const m2::DFBSpecName& name,
                       const m2::TensorParamName& borrowed,
                       uint32_t entry_size,
                       uint32_t num_entries,
                       tt::DataFormat df,
                       const tt::tt_metal::Tile& tile) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = df,
            .tile_format_metadata = tile,
            .borrowed_from = borrowed,
        };
    };
    m2::DataflowBufferSpec in0_dfb = make_dfb(IN0, T_IN, in_tile_bytes, n_shard_tiles, in_df, in_tile);
    m2::DataflowBufferSpec out_dfb = make_dfb(OUT, T_OUT, out_tile_bytes, n_shard_tiles, out_df, out_tile);

    // Reader: produces the resident input shard into IN0 (no NoC, no tensor binding needed).
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(kReaderDfb),
        .num_threads = 1,
        .dfb_bindings = {m2::ProducerOf(IN0, "in0")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config = m2::DataMovementHardwareConfig::Gen1Config{.processor = DataMovementProcessor::RISCV_1}},
    };

    // Writer: drains the resident output shard from OUT (no NoC).
    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(kWriterDfb),
        .num_threads = 1,
        .dfb_bindings = {m2::ConsumerOf(OUT, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config = m2::DataMovementHardwareConfig::Gen1Config{.processor = DataMovementProcessor::RISCV_0}},
    };

    // Compute: in0 (consumer) -> SFPU LUT eval -> out (producer). bf16 -> no fp32 dest acc.
    // The per-activation LUT (POLY or RATIONAL coefficients from the fitter CSV) is baked
    // in via -D defines (make_lut_defines); nullopt => the kernel's default sigmoid.
    m2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(kComputeDfb),
        .num_threads = 1,
        .compiler_options = {.defines = make_lut_defines(op.lut_config)},
        .dfb_bindings = {m2::ConsumerOf(IN0, "in0"), m2::ProducerOf(OUT, "out")},
        .compile_time_args = {{"num_tiles_per_cycle", num_tiles_per_cycle}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
            },
    };

    std::set<CoreRange> target_ranges;
    for (const auto& core : cores) {
        target_ranges.insert(CoreRange(core, core));
    }
    m2::NodeRangeSet target_nodes(target_ranges);

    m2::WorkUnitSpec wu{
        .name = "unary_lut_metal_v2_sharded",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = target_nodes,
    };

    m2::ProgramSpec spec{
        .name = "unary_lut_metal_v2_sharded",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in0_dfb, out_dfb},
        .tensor_parameters =
            {{.unique_id = T_IN, .spec = in.tensor_spec()}, {.unique_id = T_OUT, .spec = out.tensor_spec()}},
        .work_units = {wu},
    };

    m2::Group<m2::ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs> reader_args, writer_args, compute_args;
    reader_args.reserve(cores.size());
    writer_args.reserve(cores.size());
    compute_args.reserve(cores.size());
    for (const auto& core : cores) {
        const m2::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};
        reader_args.push_back({node, {{"num_tiles", n_shard_tiles}}});
        writer_args.push_back({node, {{"num_tiles", n_shard_tiles}}});
        compute_args.push_back({node, {{"num_tiles", n_shard_tiles}}});
    }

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE, .runtime_arg_values = std::move(compute_args)},
    };
    run_params.tensor_args.emplace(T_IN, m2::ProgramRunArgs::TensorArgument{in.mesh_tensor()});
    run_params.tensor_args.emplace(T_OUT, m2::ProgramRunArgs::TensorArgument{out.mesh_tensor()});

    return ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
        .op_owned_tensors = {},
    };
}

}  // namespace ttnn::operations::experimental::quasar::unary_lut
