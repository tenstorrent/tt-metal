// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_welford_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Metal 2.0 resource names (ProgramSpec scope). Kernels refer to these as tensor::<name>,
// dfb::<name> and args::<name>.
const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName COMPUTE{"compute"};
const m2::KernelSpecName WRITER{"writer"};

const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName OUTPUT{"output"};
const m2::TensorParamName GAMMA{"gamma"};
const m2::TensorParamName BETA{"beta"};

const m2::DFBSpecName IN{"in"};    // reader -> compute, the sliding window
const m2::DFBSpecName EPS{"eps"};  // reader -> compute, one bf16 column-scalar tile per NEO (explicit sync)
const m2::DFBSpecName GAMMA_DFB{"gamma"};
const m2::DFBSpecName BETA_DFB{"beta"};
const m2::DFBSpecName EX{"ex"};        // compute self-loop: mean column
const m2::DFBSpecName EX2{"ex2"};      // compute self-loop: var column
const m2::DFBSpecName EX2PE{"ex2pe"};  // compute self-loop: rstd column
const m2::DFBSpecName XMM{"xmm"};      // compute self-loop: normalise / affine intermediate
const m2::DFBSpecName OUT{"out"};      // compute -> writer

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/quasar/layernorm/device/kernels/";

// Thread plan: DM0 (DFB ISR) and DM1 (remapper) are reserved, leaving six DM harts; 4 readers + 2 writers
// use all six, and the 4 NEOs each take a quarter of the rows. The reader side carries 2-3x the writer's
// bytes (input twice minus the resident tail, plus gamma/beta/eps, vs. the output once), so it gets the
// extra harts. Each writer hart drains 2 NEOs in rotation. TTNN_QSR_LAYERNORM_THREADS="R,C,W" overrides
// for experiments (C % R == 0 and C % W == 0).
struct ThreadPlan {
    uint32_t reader = 4, compute = 4, writer = 2;
};

ThreadPlan thread_plan() {
    ThreadPlan plan;
    if (const char* env = std::getenv("TTNN_QSR_LAYERNORM_THREADS")) {
        unsigned r = 0, c = 0, w = 0;
        TT_FATAL(
            std::sscanf(env, "%u,%u,%u", &r, &c, &w) == 3 && r > 0 && c > 0 && w > 0,
            "TTNN_QSR_LAYERNORM_THREADS must be \"R,C,W\" with positive counts, got '{}'",
            env);
        plan = {.reader = r, .compute = c, .writer = w};
    }
    TT_FATAL(
        plan.compute % plan.reader == 0,
        "layer_norm: compute threads ({}) must be a multiple of reader threads ({}) for STRIDED rings",
        plan.compute,
        plan.reader);
    TT_FATAL(
        plan.compute % plan.writer == 0 && plan.writer <= plan.compute,
        "layer_norm: compute threads ({}) must be a multiple of writer threads ({}) for STRIDED rings",
        plan.compute,
        plan.writer);
    return plan;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts LayerNormWelfordProgramFactory::create_program_artifacts(
    const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using namespace tt::constants;

    const Tensor& input = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;
    auto* device = input.device();
    const ThreadPlan threads = thread_plan();
    const uint32_t R = threads.reader;
    const uint32_t C = threads.compute;
    const uint32_t Wr = threads.writer;

    // ---- Geometry ----
    // W is the logical width (the kernel masks the ragged last tile and scales by 1/W); Wt comes from the
    // padded width.
    const uint32_t W = input.logical_shape()[-1];
    const uint32_t W_padded = input.padded_shape()[-1];
    const uint32_t Wt = W_padded / TILE_WIDTH;
    const uint32_t num_tile_rows = input.physical_volume() / W_padded / TILE_HEIGHT;
    const float eps = operation_attributes.eps;

    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);
    // Sliding window per NEO is 2 * B tiles: half stays resident between the passes, half is the DM0 ISR
    // credit batch (two txn ids). B must divide Wt: every ring's per-row stream (in: Wt + head, gamma /
    // beta / out: Wt) is then an exact multiple of the batch, so the ISR posts every credit and the
    // end-of-kernel top-up in ~DataflowBuffer() is never needed. That matters because the destructors
    // drain the rings one at a time in reverse declaration order, and a partial tail on one ring whose
    // top-up is still pending can deadlock the drain of another ring the compute consumes interleaved.
    auto largest_divisor_up_to = [](uint32_t n, uint32_t cap) {
        for (uint32_t d = cap; d > 1; --d) {
            if (n % d == 0) {
                return d;
            }
        }
        return 1u;
    };
    const uint32_t batch = largest_divisor_up_to(Wt, 4);
    const uint32_t cb_length = 2 * batch;
    // Pass-2 block: DST tiles per acquire (fp32 dest halves the register file), at most one credit batch.
    const uint32_t block_size = std::min(batch, fp32_dest_acc_en ? 2u : 4u);
    TT_FATAL(cb_length % 2 == 0 && block_size <= cb_length / 2, "layer_norm: window / block geometry");

    // ---- Formats ----
    const tt::DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    const tt::DataFormat stats_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : in_df;
    const tt::DataFormat eps_df = tt::DataFormat::Float16_b;  // generate_bcast_col_scalar writes bf16
    const auto& tile = input.tensor_spec().tile();
    auto tile_bytes = [&](tt::DataFormat df) { return tile.get_tile_size(df); };

    // ---- Work split over the compute grid ----
    const auto grid = device->compute_with_storage_grid_size();
    const CoreRangeSet full_grid({CoreRange({0, 0}, {grid.x - 1, grid.y - 1})});
    const auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        split_work_to_cores(full_grid, num_tile_rows, true);

    // ---- ProgramSpec ----
    m2::ProgramSpec spec;
    spec.name = "quasar_layernorm_welford";

    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    if (gamma.has_value()) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = GAMMA, .spec = gamma->tensor_spec()});
    }
    if (beta.has_value()) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = BETA, .spec = beta->tensor_spec()});
    }

    // DFBs. num_entries = per-thread capacity x max(producers, consumers): STRIDED rings slice into one
    // sub-ring per consumer NEO (stride C); compute self-loops are per-NEO intra rings.
    auto dfb = [&](const m2::DFBSpecName& name, tt::DataFormat df, uint32_t capacity_per_thread, uint32_t threads) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_bytes(df),
            .num_entries = capacity_per_thread * threads,
            .data_format_metadata = df,
            .tile_format_metadata = tile,
        };
    };
    spec.dataflow_buffers.push_back(dfb(IN, in_df, cb_length, C));
    spec.dataflow_buffers.push_back(dfb(EPS, eps_df, 1, C));
    if (gamma.has_value()) {
        spec.dataflow_buffers.push_back(dfb(GAMMA_DFB, datatype_to_dataformat_converter(gamma->dtype()), cb_length, C));
    }
    if (beta.has_value()) {
        spec.dataflow_buffers.push_back(dfb(BETA_DFB, datatype_to_dataformat_converter(beta->dtype()), cb_length, C));
    }
    spec.dataflow_buffers.push_back(dfb(EX, stats_df, 1, C));
    spec.dataflow_buffers.push_back(dfb(EX2, stats_df, 1, C));
    spec.dataflow_buffers.push_back(dfb(EX2PE, stats_df, 1, C));
    spec.dataflow_buffers.push_back(dfb(XMM, stats_df, 2 * block_size, C));
    spec.dataflow_buffers.push_back(dfb(OUT, out_df, cb_length, C));

    m2::KernelSpec::CompilerOptions::Defines affine_defines;
    if (gamma.has_value()) {
        affine_defines.emplace("FUSE_GAMMA", "1");
    }
    if (beta.has_value()) {
        affine_defines.emplace("FUSE_BETA", "1");
    }

    // Reader: implicit sync on the streamed rings; eps is RISC-filled and pushed explicitly.
    m2::Group<m2::TensorBinding> reader_tensors{m2::TensorBinding{INPUT, "input"}};
    m2::Group<m2::DFBBinding> reader_dfbs{m2::ProducerOf(IN, "in"), m2::ProducerOf(EPS, "eps")};
    if (gamma.has_value()) {
        reader_tensors.push_back(m2::TensorBinding{GAMMA, "gamma"});
        reader_dfbs.push_back(m2::ProducerOf(GAMMA_DFB, "gamma"));
    }
    if (beta.has_value()) {
        reader_tensors.push_back(m2::TensorBinding{BETA, "beta"});
        reader_dfbs.push_back(m2::ProducerOf(BETA_DFB, "beta"));
    }
    auto reader_hw = ttnn::create_reader_datamovement_config();
    reader_hw.config_2xx = m2::DataMovementHardwareConfig::DataMovement2XXConfig{
        .disable_dfb_implicit_sync_for = m2::Group<m2::DFBSpecName>{EPS}};
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(KERNEL_DIR) + "dataflow/reader_layernorm_welford.cpp",
        .num_threads = R,
        .compiler_options = {.defines = affine_defines},
        .dfb_bindings = reader_dfbs,
        .tensor_bindings = reader_tensors,
        .compile_time_args = {{"Wt", Wt}, {"cb_length", cb_length}, {"block_size", block_size}, {"compute_threads", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"row_start", "NCHt", "eps"}},
        .hw_config = reader_hw,
    };

    // Compute: C NEOs.
    m2::Group<m2::DFBBinding> compute_dfbs{
        m2::ConsumerOf(IN, "in"),
        m2::ConsumerOf(EPS, "eps"),
        m2::ProducerOf(EX, "ex"),
        m2::ConsumerOf(EX, "ex"),
        m2::ProducerOf(EX2, "ex2"),
        m2::ConsumerOf(EX2, "ex2"),
        m2::ProducerOf(EX2PE, "ex2pe"),
        m2::ConsumerOf(EX2PE, "ex2pe"),
        m2::ProducerOf(XMM, "xmm"),
        m2::ConsumerOf(XMM, "xmm"),
        m2::ProducerOf(OUT, "out"),
    };
    if (gamma.has_value()) {
        compute_dfbs.push_back(m2::ConsumerOf(GAMMA_DFB, "gamma"));
    }
    if (beta.has_value()) {
        compute_dfbs.push_back(m2::ConsumerOf(BETA_DFB, "beta"));
    }
    auto compute_hw = ttnn::to_compute_hardware_config(operation_attributes.compute_kernel_config);
    // fp32 operands with fp32 dest: ValidateProgramSpec wants an explicit unpack mode; the FPU path unpacks
    // to SRC. (Quasar's transpose_tile cannot unpack fp32 to DEST, validated on the device op.)
    if (in_df == tt::DataFormat::Float32) {
        compute_hw.unpack_modes.emplace(IN, UnpackMode::UnpackToSrc);
    }
    if (stats_df == tt::DataFormat::Float32) {
        for (const auto& name : {EX, EX2, EX2PE, XMM}) {
            compute_hw.unpack_modes.emplace(name, UnpackMode::UnpackToSrc);
        }
    }
    if (gamma.has_value() && gamma->dtype() == DataType::FLOAT32) {
        compute_hw.unpack_modes.emplace(GAMMA_DFB, UnpackMode::UnpackToSrc);
    }
    if (beta.has_value() && beta->dtype() == DataType::FLOAT32) {
        compute_hw.unpack_modes.emplace(BETA_DFB, UnpackMode::UnpackToSrc);
    }
    // TEMPORARY (sim bring-up): extra include path, e.g. a stub welford LLK for Quasar JIT experiments.
    m2::KernelSpec::CompilerOptions::IncludePaths compute_includes;
    if (const char* inc = std::getenv("TTNN_QSR_LAYERNORM_EXTRA_INC")) {
        compute_includes.emplace_back(inc);
    }
    auto compute_defines = affine_defines;
    if (const char* def = std::getenv("TTNN_QSR_LAYERNORM_EXTRA_DEFINE")) {
        compute_defines.emplace(def, "1");
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(KERNEL_DIR) + "compute/layernorm_welford.cpp",
        .num_threads = C,
        .compiler_options = {.include_paths = compute_includes, .defines = compute_defines},
        .dfb_bindings = compute_dfbs,
        .compile_time_args =
            {{"Wt", Wt}, {"W", W}, {"cb_length", cb_length}, {"block_size", block_size}, {"compute_threads", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt"}},
        .hw_config = compute_hw,
    };

    // Writer: hart w drains NEOs w, w+Wr, ... in rotation (STRIDED out ring, C producers / Wr consumers).
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(KERNEL_DIR) + "dataflow/writer_layernorm_welford.cpp",
        .num_threads = Wr,
        .dfb_bindings = {m2::ConsumerOf(OUT, "out")},
        .tensor_bindings = {m2::TensorBinding{OUTPUT, "output"}},
        .compile_time_args = {{"Wt", Wt}, {"cb_length", cb_length}, {"compute_threads", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"row_start", "NCHt"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    spec.kernels = {reader, compute, writer};
    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "wu",
        .kernels = {READER, COMPUTE, WRITER},
        .target_nodes = all_cores,
    });

    // ---- ProgramRunArgs ----
    m2::ProgramRunArgs run_args;
    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};

    uint32_t curr_row = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt, true)) {
        uint32_t rows = 0;
        if (core_group_1.contains(core)) {
            rows = rows_per_core_1;
        } else if (core_group_2.contains(core)) {
            rows = rows_per_core_2;
        } else {
            TT_THROW("layer_norm: core ({}, {}) is not in either work group", core.x, core.y);
        }
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"row_start", curr_row}, {"NCHt", rows}, {"eps", std::bit_cast<uint32_t>(eps)}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"NCHt", rows}});
        m2::AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"row_start", curr_row}, {"NCHt", rows}});
        curr_row += rows;
    }
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(compute_run_args), std::move(writer_run_args)};

    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());
    if (gamma.has_value()) {
        run_args.tensor_args.emplace(GAMMA, gamma->mesh_tensor());
    }
    if (beta.has_value()) {
        run_args.tensor_args.emplace(BETA, beta->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
