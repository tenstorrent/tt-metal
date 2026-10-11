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
// ROW_MAJOR sharded (upstream: the shard borrowed as the CB, every tile_size bytes of it one tile): each node of
// the shard grid streams one input shard through the same compute kernel as raw tile-sized chunks
// (reader/writer_unary_sharded_rm_dfb.cpp, ShardView + byte offset), so an elementwise op sees every value once
// and the result lands at the same offset of the output shard. Works in place (output_tensor = input).
// Not ported: the ROW_MAJOR interleaved (row-chunked) path, borrowing L1 shards in place, sub_core_grids, and
// the special compute kernels (hardswish, logit, where_tss, ...); validate_on_program_cache_miss rejects them.

#include "unary_program_factory.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
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
constexpr const char* kReaderShardedRmSource =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary/device/kernels/dataflow/reader_unary_sharded_rm_dfb.cpp";
constexpr const char* kWriterShardedRmSource =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary/device/kernels/dataflow/writer_unary_sharded_rm_dfb.cpp";
constexpr const char* kComputeSource =
    "ttnn/cpp/ttnn/operations/experimental/quasar/unary/device/kernels/compute/eltwise_sfpu_dfb.cpp";

// The upstream factory's interleaved CB depth: double-buffered single-tile pages.
constexpr uint32_t kNumDfbEntries = 2;

// Per-node work: TILE -> a contiguous run of tile ids; ROW_MAJOR sharded -> one shard, cut into num_units
// chunks of one DFB entry (the last one last_chunk_bytes long).
struct NodeWork {
    tt::tt_metal::CoreCoord core;
    uint32_t num_units = 0;
    uint32_t start_id = 0;  // TILE: first tile id; ROW_MAJOR sharded: shard id
    uint32_t last_chunk_bytes = 0;
};

// ROW_MAJOR sharded: node i of the shard grid (in shard orientation order, the order the shards are placed in)
// takes shard i, normally its own. The accessor resolves the shard's node, so the result does not depend on that,
// only the NoC traffic does. One shard = shard_shape[0] pages of
// aligned_page_size bytes, contiguous in L1; a partial edge shard is still allocated (and processed) in full.
std::vector<NodeWork> split_row_major_shards(const Tensor& input, uint32_t chunk_bytes) {
    const auto& shard_spec = input.shard_spec().value();
    const auto& shape = input.padded_shape();
    const uint32_t width = shape[-1];
    const uint32_t height = width == 0 ? 0 : static_cast<uint32_t>(shape.volume() / width);
    const uint32_t num_shards = tt::div_up(height, shard_spec.shape[0]) * tt::div_up(width, shard_spec.shape[1]);
    const auto cores = tt::tt_metal::corerange_to_cores(
        shard_spec.grid, std::nullopt, shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    TT_FATAL(
        num_shards <= cores.size(),
        "Quasar unary: {} ROW_MAJOR shards do not fit the {}-node shard grid",
        num_shards,
        cores.size());

    const uint32_t shard_bytes = shard_spec.shape[0] * input.buffer()->aligned_page_size();
    const uint32_t num_chunks = tt::div_up(shard_bytes, chunk_bytes);
    const uint32_t last_chunk_bytes = shard_bytes - ((num_chunks - 1) * chunk_bytes);
    std::vector<NodeWork> work;
    work.reserve(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        work.push_back(
            NodeWork{.core = cores[i], .num_units = num_chunks, .start_id = i, .last_chunk_bytes = last_chunk_bytes});
    }
    return work;
}

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
    const bool row_major_sharded = input.layout() == tt::tt_metal::Layout::ROW_MAJOR;

    // ---- Work split ----
    std::vector<CMAKE_UNIQUE_NAMESPACE::NodeWork> work;
    if (row_major_sharded) {
        TT_FATAL(
            input.buffer()->aligned_page_size() == output.buffer()->aligned_page_size(),
            "Quasar unary: ROW_MAJOR input and output shard rows differ ({} vs {} B)",
            input.buffer()->aligned_page_size(),
            output.buffer()->aligned_page_size());
        work = CMAKE_UNIQUE_NAMESPACE::split_row_major_shards(input, tt::tile_size(input_df));
    } else {
        // Output tiles spread row-wise over the worker grid (upstream TILE path).
        const uint32_t num_tiles = output.physical_volume() / tt::constants::TILE_HW;
        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(args.worker_grid, num_tiles, /*row_wise=*/true);
        (void)num_cores;
        uint32_t start_id = 0;
        for (const auto& core : tt::tt_metal::corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true)) {
            uint32_t num_tiles_per_core = 0;
            if (core_group_1.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                TT_THROW("Quasar unary: core {} is in neither work group", core.str());
            }
            work.push_back(
                CMAKE_UNIQUE_NAMESPACE::NodeWork{.core = core, .num_units = num_tiles_per_core, .start_id = start_id});
            start_id += num_tiles_per_core;
        }
    }
    TT_FATAL(!work.empty(), "Quasar unary: no cores to run on");
    std::vector<CoreCoord> cores;
    cores.reserve(work.size());
    for (const auto& w : work) {
        cores.push_back(w.core);
    }
    const auto all_cores = CoreRangeSet(ttsl::Span<const CoreCoord>(cores));

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
    // Dataflow runtime args: TILE {num_pages, start_id}; ROW_MAJOR sharded {shard_id, num_chunks, last_chunk_bytes}.
    const std::vector<std::string> dm_arg_names =
        row_major_sharded ? std::vector<std::string>{"shard_id", "num_chunks", "last_chunk_bytes"}
                          : std::vector<std::string>{"num_pages", "start_id"};
    const m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            row_major_sharded ? CMAKE_UNIQUE_NAMESPACE::kReaderShardedRmSource : CMAKE_UNIQUE_NAMESPACE::kReaderSource),
        .num_threads = 1,
        .dfb_bindings = {m2::ProducerOf(IN_DFB, "in")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = dm_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    const m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            row_major_sharded ? CMAKE_UNIQUE_NAMESPACE::kWriterShardedRmSource : CMAKE_UNIQUE_NAMESPACE::kWriterSource),
        .num_threads = 1,
        .dfb_bindings = {m2::ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = dm_arg_names},
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

    // ---- Per-node runtime args ----
    m2::ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    m2::ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (const auto& w : work) {
        if (row_major_sharded) {
            for (auto* run_args : {&reader_run_args, &writer_run_args}) {
                m2::AddRuntimeArgsForNode(
                    run_args->runtime_arg_values,
                    w.core,
                    {{"shard_id", w.start_id}, {"num_chunks", w.num_units}, {"last_chunk_bytes", w.last_chunk_bytes}});
            }
        } else {
            m2::AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values, w.core, {{"num_pages", w.num_units}, {"start_id", w.start_id}});
            m2::AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values, w.core, {{"num_pages", w.num_units}, {"start_id", w.start_id}});
        }
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, w.core, {{"num_tiles", w.num_units}});
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
