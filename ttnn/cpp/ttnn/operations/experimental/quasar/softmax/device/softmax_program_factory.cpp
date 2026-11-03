// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_program_factory.hpp"

#include <algorithm>
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

const m2::DFBSpecName IN{"in"};          // reader -> compute, the row streamed once per pass
const m2::DFBSpecName SCALER{"scaler"};  // reader -> compute, one reduce-scaler tile per NEO (explicit sync)
const m2::DFBSpecName ROWMAX{"rowmax"};  // compute self-loop: per-row max, column 0
const m2::DFBSpecName EXPS{"exps"};      // compute self-loop: exp(x - max) block
const m2::DFBSpecName PSUM{"psum"};      // compute self-loop: row-sum of one block, column 0
const m2::DFBSpecName ROWSUM{"rowsum"};  // compute self-loop: running row-sum, column 0
const m2::DFBSpecName RSUM{"rsum"};      // compute self-loop: 1 / row-sum, column 0
const m2::DFBSpecName OUT{"out"};        // compute -> writer

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/quasar/softmax/device/kernels/";

// Thread plan: DM0 (DFB ISR) and DM1 (remapper) are reserved, leaving six DM harts; 4 readers + 2 writers
// use all six, and the 4 NEOs each take a quarter of the rows. The reader streams every row two or three
// times (max pass, sum pass, normalise pass) while the writer stores it once, so the reader side gets the
// extra harts. Each writer hart drains 2 NEOs in rotation.
// The plan is an op attribute (ThreadPlan, see common/thread_plan.hpp); the host entry reads the
// TTNN_QSR_SOFTMAX_THREADS override.

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts SoftmaxProgramFactory::create_program_artifacts(
    const SoftmaxParams& operation_attributes, const SoftmaxInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using namespace tt::constants;

    const Tensor& input = tensor_args.input;
    const Tensor& output = tensor_return_value;
    auto* device = input.device();
    const ThreadPlan& threads = operation_attributes.threads;
    const uint32_t R = threads.reader;
    const uint32_t C = threads.compute;
    const uint32_t Wr = threads.writer;

    // ---- Geometry ----
    const uint32_t W_padded = input.padded_shape()[-1];
    const uint32_t Wt = W_padded / TILE_WIDTH;
    const uint32_t num_tile_rows = input.physical_volume() / W_padded / TILE_HEIGHT;
    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);

    // Per-NEO window 2 * B with B a divisor of Wt: every ring's per-row stream (in: 2 or 3 x Wt, out: Wt)
    // is then a whole number of DM0 ISR credit batches, so the ~DataflowBuffer drains (reverse declaration
    // order, at kernel exit) never wait on a partial batch the ISR will not complete.
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
    // Tiles per DST acquire in the exp / normalise blocks (fp32 dest halves the register file). A divisor
    // of Wt, so every block is full: a pack batch must never straddle the end of its ring (the packer
    // writes a batch linearly), and with full blocks the 2 * block_size exps ring and the 2 * batch out
    // ring are always entered on a block boundary.
    const uint32_t block_size = largest_divisor_up_to(Wt, fp32_dest_acc_en ? 2u : 4u);

    // ---- Formats ----
    const tt::DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    const tt::DataFormat stats_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : in_df;
    const tt::DataFormat scaler_df = tt::DataFormat::Float16_b;  // the reader fills a bf16 tile
    const auto& tile = input.tensor_spec().tile();
    auto tile_bytes = [&](tt::DataFormat df) { return tile.get_tile_size(df); };

    // ---- Work split over the compute grid ----
    const auto grid = device->compute_with_storage_grid_size();
    const CoreRangeSet full_grid({CoreRange({0, 0}, {grid.x - 1, grid.y - 1})});
    const auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        split_work_to_cores(full_grid, num_tile_rows, true);

    // ---- ProgramSpec ----
    m2::ProgramSpec spec;
    spec.name = "quasar_softmax";

    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    // num_entries = per-thread capacity x NEOs: STRIDED rings slice into one sub-ring per consumer NEO;
    // compute self-loops are per-NEO intra rings.
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
    spec.dataflow_buffers.push_back(dfb(SCALER, scaler_df, 1));
    spec.dataflow_buffers.push_back(dfb(ROWMAX, stats_df, 1));
    spec.dataflow_buffers.push_back(dfb(EXPS, stats_df, 2 * block_size));
    spec.dataflow_buffers.push_back(dfb(PSUM, stats_df, 2));
    spec.dataflow_buffers.push_back(dfb(ROWSUM, stats_df, 2));
    spec.dataflow_buffers.push_back(dfb(RSUM, stats_df, 1));
    spec.dataflow_buffers.push_back(dfb(OUT, out_df, cb_length));

    m2::KernelSpec::CompilerOptions::Defines defines;
    if (operation_attributes.numeric_stable) {
        defines.emplace("NUMERIC_STABLE", "1");
    }

    // Reader: implicit sync on dfb::in; the scaler tile is RISC-filled and pushed explicitly.
    auto reader_hw = ttnn::create_reader_datamovement_config();
    reader_hw.config_2xx = m2::DataMovementHardwareConfig::DataMovement2XXConfig{
        .disable_dfb_implicit_sync_for = m2::Group<m2::DFBSpecName>{SCALER}};
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(KERNEL_DIR) + "dataflow/reader_softmax.cpp",
        .num_threads = R,
        .compiler_options = {.defines = defines},
        .dfb_bindings = {m2::ProducerOf(IN, "in"), m2::ProducerOf(SCALER, "scaler")},
        .tensor_bindings = {m2::TensorBinding{INPUT, "input"}},
        .compile_time_args = {{"Wt", Wt}, {"compute_threads", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"row_start", "NCHt"}},
        .hw_config = reader_hw,
    };

    // Compute: C NEOs.
    auto compute_hw = ttnn::to_compute_hardware_config(operation_attributes.compute_kernel_config);
    if (stats_df == tt::DataFormat::Float32) {
        for (const auto& name : {ROWMAX, EXPS, PSUM, ROWSUM, RSUM}) {
            compute_hw.unpack_modes.emplace(name, UnpackMode::UnpackToSrc);
        }
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(KERNEL_DIR) + "compute/softmax.cpp",
        .num_threads = C,
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {m2::ConsumerOf(IN, "in"),
             m2::ConsumerOf(SCALER, "scaler"),
             m2::ProducerOf(ROWMAX, "rowmax"),
             m2::ConsumerOf(ROWMAX, "rowmax"),
             m2::ProducerOf(EXPS, "exps"),
             m2::ConsumerOf(EXPS, "exps"),
             m2::ProducerOf(PSUM, "psum"),
             m2::ConsumerOf(PSUM, "psum"),
             m2::ProducerOf(ROWSUM, "rowsum"),
             m2::ConsumerOf(ROWSUM, "rowsum"),
             m2::ProducerOf(RSUM, "rsum"),
             m2::ConsumerOf(RSUM, "rsum"),
             m2::ProducerOf(OUT, "out")},
        .compile_time_args = {{"Wt", Wt}, {"block_size", block_size}, {"compute_threads", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt"}},
        .hw_config = compute_hw,
    };

    // Writer: hart w drains NEOs w, w+Wr, ... in rotation (STRIDED out ring, C producers / Wr consumers).
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(KERNEL_DIR) + "dataflow/writer_softmax.cpp",
        .num_threads = Wr,
        .dfb_bindings = {m2::ConsumerOf(OUT, "out")},
        .tensor_bindings = {m2::TensorBinding{OUTPUT, "output"}},
        .compile_time_args = {{"Wt", Wt}, {"compute_threads", C}},
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
            TT_THROW("softmax: core ({}, {}) is not in either work group", core.x, core.y);
        }
        m2::AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"row_start", curr_row}, {"NCHt", rows}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"NCHt", rows}});
        m2::AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"row_start", curr_row}, {"NCHt", rows}});
        curr_row += rows;
    }
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(compute_run_args), std::move(writer_run_args)};
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
