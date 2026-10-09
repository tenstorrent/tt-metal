// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) port of the TILE path of the WH/BH unary program factory
// (ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp). It keeps that factory's
// op-chain defines (get_block_defines: SFPU_OP_CHAIN_0 + the per-family include macro), compute config
// (HiFi4, precise SFPU, fp32 dest when requested, UnpackToDest for preserve_fp32_precision) and work split
// (split_work_to_cores over the worker grid, row-wise), and differs from it in the Metal 2.0 translation:
//   - CBDescriptor c_0 / c_2 -> DataflowBufferSpec "in" / "out", 2 tile entries each.
//   - Buffer-address runtime args + TensorAccessorArgs -> TensorParameter + TensorBinding
//     (TensorAccessor(tensor::src / tensor::dst) in the kernels). The accessor is sharding-aware, so an
//     interleaved or sharded operand is read / written over the NoC with the same kernels.
//   - Positional runtime args -> named args; per-core dummy args on idle cores -> WorkUnitSpec::target_nodes
//     scoped to the active cores.
//   - Quasar: num_threads 1 for the reader, the compute and the writer (one DM thread each way and one Neo
//     per node), and explicit DFB sync on the DM side (disable_dfb_implicit_sync_for_all), so the kernels'
//     reserve_back/push_back and wait_front/pop_front stay authoritative.
// Not ported: the ROW_MAJOR (row-chunked) path, borrowing L1 shards in place, sub_core_grids, and the
// special compute kernels (hardswish, logit, where_tss, ...); validate_on_program_cache_miss rejects them.

#include "unary_program_factory.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr const char* kReaderSource =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary/device/kernels/dataflow/reader_unary_dfb.cpp";
constexpr const char* kWriterSource =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary/device/kernels/dataflow/writer_unary_dfb.cpp";
constexpr const char* kComputeSource =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary/device/kernels/compute/eltwise_sfpu_dfb.cpp";

// The upstream factory's interleaved CB depth: double-buffered single-tile pages.
constexpr uint32_t kNumDfbEntries = 2;

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts UnaryProgramFactory::create_program_artifacts(
    const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output) {
    namespace m2 = tt::tt_metal::experimental;
    namespace unary_utils = ttnn::operations::unary::utils;
    using tt::tt_metal::CoreCoord;
    using tt::tt_metal::CoreRangeSet;

    const Tensor& input = tensor_args.input;

    const tt::DataFormat input_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    // ---- Work split: output tiles spread row-wise over the worker grid (upstream TILE path) ----
    const uint32_t num_tiles = output.physical_volume() / tt::constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(args.worker_grid, num_tiles, /*row_wise=*/true);
    (void)num_cores;
    const std::vector<CoreCoord> cores = tt::tt_metal::corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    TT_FATAL(!cores.empty(), "Quasar unary: no cores to run on ({} tiles)", num_tiles);

    // ---- Resource names ----
    const m2::DFBSpecName IN_DFB{"in"};    // legacy CBIndex::c_0
    const m2::DFBSpecName OUT_DFB{"out"};  // legacy CBIndex::c_2
    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName OUTPUT{"output"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = tt::tile_size(input_df),
        .num_entries = CMAKE_UNIQUE_NAMESPACE::kNumDfbEntries,
        .data_format_metadata = input_df,
    };
    const m2::DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = tt::tile_size(output_df),
        .num_entries = CMAKE_UNIQUE_NAMESPACE::kNumDfbEntries,
        .data_format_metadata = output_df,
    };

    // ---- Kernels ----
    const m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(CMAKE_UNIQUE_NAMESPACE::kReaderSource),
        .num_threads = 1,
        .dfb_bindings = {m2::ProducerOf(IN_DFB, "in")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    const m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(CMAKE_UNIQUE_NAMESPACE::kWriterSource),
        .num_threads = 1,
        .dfb_bindings = {m2::ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    // SFPU_OP_CHAIN_0 (init + call per chained op) and the include macro of each op's family, exactly as the
    // upstream factory builds them, plus the INP_* dtype tag some SFPU kernels select their algorithm on.
    std::map<std::string, std::string> unary_defines =
        unary_utils::get_block_defines(args.op_chain, "0", "0", input.dtype());
    unary_utils::add_input_dtype_defines(input.dtype(), unary_defines);
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [name, value] : unary_defines) {
        compute_defines.emplace(name, value);
    }

    // Legacy: unpack_to_dest_mode[c_0] = UnpackToDestFp32 when preserve_fp32_precision. Metal 2.0 also wants an
    // explicit entry for a Float32 DFB consumed with a 32-bit dest, which legacy defaulted to UnpackToSrc.
    m2::ComputeHardwareConfig::ComputeUnpackModes unpack_modes;
    if (args.preserve_fp32_precision) {
        unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
    } else if (args.fp32_dest_acc_en && input_df == tt::DataFormat::Float32) {
        unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
    }
    // Legacy ComputeConfigDescriptor: HiFi4, math_approx_mode = false, fp32_dest_acc_en, dst_full_sync_en = false
    // (double-buffered dest, the Metal 2.0 default). Quasar has no BFP pack precision knob, so config_1xx is set
    // only off Quasar.
    m2::ComputeHardwareConfig compute_hw{
        .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .sfpu_precision_mode = tt::tt_metal::Precision::Precise,
        .enable_32_bit_dest = args.fp32_dest_acc_en,
        .unpack_modes = std::move(unpack_modes),
    };
    if (input.device()->arch() != tt::ARCH::QUASAR) {
        compute_hw.config_1xx = m2::ComputeHardwareConfig::Compute1XXConfig{};
    }

    const m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(CMAKE_UNIQUE_NAMESPACE::kComputeSource),
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = {m2::ConsumerOf(IN_DFB, "in"), m2::ProducerOf(OUT_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = compute_hw,
    };

    // ---- Per-node runtime args: each active node walks a contiguous run of output tile ids ----
    m2::ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    m2::ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    uint32_t start_id = 0;
    for (const auto& core : cores) {
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Quasar unary: core {} is in neither work group", core.str());
        }
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", start_id}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", start_id}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}});
        start_id += num_tiles_per_core;
    }

    m2::ProgramSpec spec{
        .name = "qsr_unary",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters =
            {m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
             m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{
            .name = "qsr_unary", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = CoreRangeSet(all_cores)}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
