// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_program_factory.hpp"

#include <algorithm>
#include <bit>
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

const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName COMPUTE{"compute"};
const m2::KernelSpecName WRITER{"writer"};

const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName OUTPUT{"output"};
const m2::TensorParamName MEAN{"mean"};
const m2::TensorParamName VAR{"var"};
const m2::TensorParamName WEIGHT{"weight"};
const m2::TensorParamName BIAS{"bias"};

const m2::DFBSpecName IN{"in"};              // reader -> compute, the plane's tiles
const m2::DFBSpecName EPS{"eps"};            // reader -> compute, one bf16 scalar tile per NEO (explicit sync)
const m2::DFBSpecName MEAN_DFB{"mean"};      // reader -> compute, one tile per plane (channel c)
const m2::DFBSpecName VAR_DFB{"var"};        // reader -> compute, one tile per plane
const m2::DFBSpecName WEIGHT_DFB{"weight"};  // reader -> compute, one tile per plane (if FUSE_WEIGHT)
const m2::DFBSpecName BIAS_DFB{"bias"};      // reader -> compute, one tile per plane (if FUSE_BIAS)
const m2::DFBSpecName RSTD{"rstd"};          // compute self-loop: 1 / sqrt(var + eps) at (0, 0)
const m2::DFBSpecName SCALE{"scale"};        // compute self-loop: rstd * weight at (0, 0)
const m2::DFBSpecName TMP{"tmp"};            // compute self-loop: mean * scale at (0, 0)
const m2::DFBSpecName SHIFT{"shift"};        // compute self-loop: bias - mean * scale at (0, 0)
const m2::DFBSpecName XMM{"xmm"};            // compute self-loop: x * scale block
const m2::DFBSpecName OUT{"out"};            // compute -> writer

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/quasar/batch_norm/device/kernels/";

// Thread plan: DM0 (DFB ISR) and DM1 (remapper) are reserved, leaving six DM harts; 4 readers + 2 writers
// use all six, and the 4 NEOs each take a quarter of the planes. The reader moves the input plus four
// parameter tiles per plane while the writer stores the output once, so the reader side gets the extra
// harts. Each writer hart drains 2 NEOs in rotation.
// The plan is an op attribute (ThreadPlan, see common/thread_plan.hpp); the host entry reads the
// TTNN_QSR_BATCH_NORM_THREADS override.

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts BatchNormProgramFactory::create_program_artifacts(
    const BatchNormParams& operation_attributes, const BatchNormInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using namespace tt::constants;

    const Tensor& input = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const Tensor& mean = tensor_args.running_mean;
    const Tensor& var = tensor_args.running_var;
    const auto& weight = tensor_args.weight;
    const auto& bias = tensor_args.bias;
    auto* device = input.device();
    const ThreadPlan& threads = operation_attributes.threads;
    const uint32_t R = threads.reader;
    const uint32_t C = threads.compute;
    const uint32_t Wr = threads.writer;

    // ---- Geometry ----
    // The unit of work is one (n, c) plane of HtWt tiles; planes are numbered n * channels + c, so the
    // input / output page of tile t of plane p is p * HtWt + t and the parameter page is c = p % channels.
    const auto& padded = input.padded_shape();
    const uint32_t N = padded[0];
    const uint32_t channels = padded[1];
    const uint32_t HtWt = (padded[2] / TILE_HEIGHT) * (padded[3] / TILE_WIDTH);
    const uint32_t num_planes = N * channels;
    const float eps = operation_attributes.eps;
    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);

    // Per-NEO window 2 * B with B a divisor of HtWt: the in / out streams per plane are then whole DM0 ISR
    // credit batches; the one-tile-per-plane parameter rings use a window of 2 (batch 1). So no ring ever
    // ends on a partial batch and the ~DataflowBuffer drains cannot deadlock on each other.
    auto largest_divisor_up_to = [](uint32_t n, uint32_t cap) {
        for (uint32_t d = cap; d > 1; --d) {
            if (n % d == 0) {
                return d;
            }
        }
        return 1u;
    };
    const uint32_t batch = largest_divisor_up_to(HtWt, 4);
    const uint32_t cb_length = 2 * batch;
    // Tiles per DST acquire, a divisor of HtWt so every block is full: a pack batch must never straddle
    // the end of its ring (the packer writes a batch linearly).
    const uint32_t block_size = largest_divisor_up_to(HtWt, fp32_dest_acc_en ? 2u : 4u);

    // ---- Formats ----
    const tt::DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    const tt::DataFormat param_df = datatype_to_dataformat_converter(mean.dtype());
    const tt::DataFormat stats_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : in_df;
    const tt::DataFormat eps_df = tt::DataFormat::Float16_b;  // generate_bcast_unary_scalar writes bf16
    const auto& tile = input.tensor_spec().tile();
    auto tile_bytes = [&](tt::DataFormat df) { return tile.get_tile_size(df); };

    // ---- Work split over the compute grid (planes) ----
    const auto grid = device->compute_with_storage_grid_size();
    const CoreRangeSet full_grid({CoreRange({0, 0}, {grid.x - 1, grid.y - 1})});
    const auto [num_cores, all_cores, core_group_1, core_group_2, planes_per_core_1, planes_per_core_2] =
        split_work_to_cores(full_grid, num_planes, true);

    // ---- ProgramSpec ----
    m2::ProgramSpec spec;
    spec.name = "quasar_batch_norm";

    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = MEAN, .spec = mean.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = VAR, .spec = var.tensor_spec()});
    if (weight.has_value()) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = WEIGHT, .spec = weight->tensor_spec()});
    }
    if (bias.has_value()) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = BIAS, .spec = bias->tensor_spec()});
    }

    auto dfb = [&](const m2::DFBSpecName& name, tt::DataFormat df, uint32_t capacity_per_thread) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_bytes(df),
            .num_entries = capacity_per_thread * C,
            .data_format_metadata = df,
            .tile_format_metadata = tile,
        };
    };
    spec.dataflow_buffers.push_back(dfb(IN, in_df, cb_length));
    spec.dataflow_buffers.push_back(dfb(EPS, eps_df, 1));
    spec.dataflow_buffers.push_back(dfb(MEAN_DFB, param_df, 2));
    spec.dataflow_buffers.push_back(dfb(VAR_DFB, param_df, 2));
    if (weight.has_value()) {
        spec.dataflow_buffers.push_back(dfb(WEIGHT_DFB, param_df, 2));
    }
    if (bias.has_value()) {
        spec.dataflow_buffers.push_back(dfb(BIAS_DFB, param_df, 2));
    }
    spec.dataflow_buffers.push_back(dfb(RSTD, stats_df, 1));
    spec.dataflow_buffers.push_back(dfb(SCALE, stats_df, 1));
    spec.dataflow_buffers.push_back(dfb(TMP, stats_df, 1));
    spec.dataflow_buffers.push_back(dfb(SHIFT, stats_df, 1));
    spec.dataflow_buffers.push_back(dfb(XMM, stats_df, 2 * block_size));
    spec.dataflow_buffers.push_back(dfb(OUT, out_df, cb_length));

    m2::KernelSpec::CompilerOptions::Defines affine_defines;
    if (weight.has_value()) {
        affine_defines.emplace("FUSE_WEIGHT", "1");
    }
    if (bias.has_value()) {
        affine_defines.emplace("FUSE_BIAS", "1");
    }

    // Reader: implicit sync on the streamed rings; eps is RISC-filled and pushed explicitly.
    m2::Group<m2::TensorBinding> reader_tensors{
        m2::TensorBinding{INPUT, "input"}, m2::TensorBinding{MEAN, "mean"}, m2::TensorBinding{VAR, "var"}};
    m2::Group<m2::DFBBinding> reader_dfbs{
        m2::ProducerOf(IN, "in"),
        m2::ProducerOf(EPS, "eps"),
        m2::ProducerOf(MEAN_DFB, "mean"),
        m2::ProducerOf(VAR_DFB, "var")};
    if (weight.has_value()) {
        reader_tensors.push_back(m2::TensorBinding{WEIGHT, "weight"});
        reader_dfbs.push_back(m2::ProducerOf(WEIGHT_DFB, "weight"));
    }
    if (bias.has_value()) {
        reader_tensors.push_back(m2::TensorBinding{BIAS, "bias"});
        reader_dfbs.push_back(m2::ProducerOf(BIAS_DFB, "bias"));
    }
    auto reader_hw = ttnn::create_reader_datamovement_config();
    reader_hw.config_2xx = m2::DataMovementHardwareConfig::DataMovement2XXConfig{
        .disable_dfb_implicit_sync_for = m2::Group<m2::DFBSpecName>{EPS}};
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(KERNEL_DIR) + "dataflow/reader_batch_norm.cpp",
        .num_threads = R,
        .compiler_options = {.defines = affine_defines},
        .dfb_bindings = reader_dfbs,
        .tensor_bindings = reader_tensors,
        .compile_time_args = {{"HtWt", HtWt}, {"channels", channels}, {"compute_threads", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"plane_start", "num_planes", "eps"}},
        .hw_config = reader_hw,
    };

    // Compute: C NEOs.
    m2::Group<m2::DFBBinding> compute_dfbs{
        m2::ConsumerOf(IN, "in"),
        m2::ConsumerOf(EPS, "eps"),
        m2::ConsumerOf(MEAN_DFB, "mean"),
        m2::ConsumerOf(VAR_DFB, "var"),
        m2::ProducerOf(RSTD, "rstd"),
        m2::ConsumerOf(RSTD, "rstd"),
        m2::ProducerOf(SCALE, "scale"),
        m2::ConsumerOf(SCALE, "scale"),
        m2::ProducerOf(TMP, "tmp"),
        m2::ConsumerOf(TMP, "tmp"),
        m2::ProducerOf(SHIFT, "shift"),
        m2::ConsumerOf(SHIFT, "shift"),
        m2::ProducerOf(XMM, "xmm"),
        m2::ConsumerOf(XMM, "xmm"),
        m2::ProducerOf(OUT, "out"),
    };
    if (weight.has_value()) {
        compute_dfbs.push_back(m2::ConsumerOf(WEIGHT_DFB, "weight"));
    }
    if (bias.has_value()) {
        compute_dfbs.push_back(m2::ConsumerOf(BIAS_DFB, "bias"));
    }
    auto compute_hw = ttnn::to_compute_hardware_config(operation_attributes.compute_kernel_config);
    if (stats_df == tt::DataFormat::Float32) {
        for (const auto& name : {RSTD, SCALE, TMP, SHIFT, XMM}) {
            compute_hw.unpack_modes.emplace(name, UnpackMode::UnpackToSrc);
        }
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(KERNEL_DIR) + "compute/batch_norm.cpp",
        .num_threads = C,
        .compiler_options = {.defines = affine_defines},
        .dfb_bindings = compute_dfbs,
        .compile_time_args = {{"HtWt", HtWt}, {"block_size", block_size}, {"compute_threads", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_planes"}},
        .hw_config = compute_hw,
    };

    // Writer: hart w drains NEOs w, w+Wr, ... in rotation (STRIDED out ring, C producers / Wr consumers).
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(KERNEL_DIR) + "dataflow/writer_batch_norm.cpp",
        .num_threads = Wr,
        .dfb_bindings = {m2::ConsumerOf(OUT, "out")},
        .tensor_bindings = {m2::TensorBinding{OUTPUT, "output"}},
        .compile_time_args = {{"HtWt", HtWt}, {"compute_threads", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"plane_start", "num_planes"}},
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

    uint32_t curr_plane = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt, true)) {
        uint32_t planes = 0;
        if (core_group_1.contains(core)) {
            planes = planes_per_core_1;
        } else if (core_group_2.contains(core)) {
            planes = planes_per_core_2;
        } else {
            TT_THROW("batch_norm: core ({}, {}) is not in either work group", core.x, core.y);
        }
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"plane_start", curr_plane}, {"num_planes", planes}, {"eps", std::bit_cast<uint32_t>(eps)}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_planes", planes}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"plane_start", curr_plane}, {"num_planes", planes}});
        curr_plane += planes;
    }
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(compute_run_args), std::move(writer_run_args)};
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());
    run_args.tensor_args.emplace(MEAN, mean.mesh_tensor());
    run_args.tensor_args.emplace(VAR, var.mesh_tensor());
    if (weight.has_value()) {
        run_args.tensor_args.emplace(WEIGHT, weight->mesh_tensor());
    }
    if (bias.has_value()) {
        run_args.tensor_args.emplace(BIAS, bias->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
