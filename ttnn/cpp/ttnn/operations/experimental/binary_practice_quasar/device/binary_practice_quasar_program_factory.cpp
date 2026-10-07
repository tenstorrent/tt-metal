// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <set>
#include <tuple>
#include <vector>

#include "binary_practice_quasar_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::binary_practice {

namespace m2 = tt::tt_metal::experimental;

namespace {
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/binary_practice_quasar/device/kernels/";

// Tiles in one shard, rounded up to whole tiles. Every node of a sharded tensor holds a full shard in L1.
uint32_t shard_tiles(const Tensor& t) {
    const auto& shard_shape = t.shard_spec()->shape;
    const tt::tt_metal::Tile tile = t.tensor_spec().tile();
    return (tt::round_up(shard_shape[0], tile.get_height()) / tile.get_height()) *
           (tt::round_up(shard_shape[1], tile.get_width()) / tile.get_width());
}

// Interleaved: a two-entry ring in L1 (the producer fills one tile while the consumer works on the other).
// Sharded: the DFB borrows the tensor's own shard, so it is the shard and holds all of its tiles.
m2::DataflowBufferSpec make_dfb(const m2::DFBSpecName& name, const Tensor& t, const m2::TensorParamName& tensor) {
    const tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(t.dtype());
    const tt::tt_metal::Tile tile = t.tensor_spec().tile();
    m2::DataflowBufferSpec dfb{
        .unique_id = name,
        .entry_size = static_cast<uint32_t>(tile.get_tile_size(df)),
        .num_entries = t.is_sharded() ? shard_tiles(t) : 2u,
        .data_format_metadata = df,
        .tile_format_metadata = tile,
    };
    if (t.is_sharded()) {
        dfb.borrowed_from = tensor;
    }
    return dfb;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts BinaryPracticeQuasarProgramFactory::create_program_artifacts(
    const BinaryPracticeQuasarParams&, const BinaryPracticeQuasarInputs& inputs, Tensor& output) {
    const Tensor& a = inputs.a;
    const Tensor& b = inputs.b;

    // Names that tie the pieces together: kernels see them as tensor::a, dfb::a, args::num_tiles, ...
    const m2::TensorParamName T_A{"a"}, T_B{"b"}, T_OUT{"out"};
    const m2::DFBSpecName DFB_A{"a"}, DFB_B{"b"}, DFB_OUT{"out"};
    const m2::KernelSpecName READER{"reader"}, COMPUTE{"compute"}, WRITER{"writer"};

    // Validation guarantees all or nothing: a sharded means b and the output are sharded the same way.
    const bool sharded = a.is_sharded();

    // --- Which nodes run, and how many tiles each. ---
    // Interleaved: split the tiles across the node grid; node k gets tiles [start, start + n).
    // Sharded: every node of the shard grid processes the shard it holds; start is unused.
    std::vector<CoreCoord> cores;
    std::function<uint32_t(const CoreCoord&)> tiles_on;
    CoreRangeSet group_1;
    uint32_t tiles_per_node_1 = 0, tiles_per_node_2 = 0;
    if (sharded) {
        const auto& shard_spec = *a.shard_spec();
        cores = corerange_to_cores(
            shard_spec.grid, std::nullopt, shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        const uint32_t n = shard_tiles(a);
        tiles_on = [n](const CoreCoord&) { return n; };
    } else {
        const uint32_t num_tiles = a.physical_volume() / tt::constants::TILE_HW;
        const auto grid = a.device()->compute_with_storage_grid_size();
        CoreRangeSet all_nodes, group_2;
        std::tie(std::ignore, all_nodes, group_1, group_2, tiles_per_node_1, tiles_per_node_2) =
            tt::tt_metal::split_work_to_cores(grid, num_tiles, /*row_wise=*/true);
        cores = corerange_to_cores(all_nodes, std::nullopt, /*row_wise=*/true);
        tiles_on = [&](const CoreCoord& core) { return group_1.contains(core) ? tiles_per_node_1 : tiles_per_node_2; };
    }

    m2::KernelRunArgs::RuntimeArgValues reader_args, compute_args, writer_args;
    std::set<CoreRange> target_ranges;
    uint32_t start_tile_id = 0;
    for (const CoreCoord& core : cores) {
        const uint32_t n = tiles_on(core);
        const m2::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};
        target_ranges.insert(CoreRange(core, core));

        reader_args["start_tile_id"][node] = start_tile_id;
        reader_args["num_tiles"][node] = n;
        compute_args["num_tiles"][node] = n;
        writer_args["start_tile_id"][node] = start_tile_id;
        writer_args["num_tiles"][node] = n;
        start_tile_id += n;
    }

    // --- Kernels. Bindings say which DFB each kernel produces/consumes and which tensors it addresses. ---
    // Sharded: the reader and writer touch no tensor over the NoC (the DFBs are the shards), so they get
    // no tensor bindings, and SHARDED compiles in their publish/drain-only branch.
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(kKernelDir) + "reader.cpp",
        .dfb_bindings = {m2::ProducerOf(DFB_A, "a"), m2::ProducerOf(DFB_B, "b")},
        .runtime_arg_schema = {.runtime_arg_names = {"start_tile_id", "num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(kKernelDir) + "compute.cpp",
        .dfb_bindings = {m2::ConsumerOf(DFB_A, "a"), m2::ConsumerOf(DFB_B, "b"), m2::ProducerOf(DFB_OUT, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::to_compute_hardware_config(ttnn::ComputeKernelConfig{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = false}),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(kKernelDir) + "writer.cpp",
        .dfb_bindings = {m2::ConsumerOf(DFB_OUT, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"start_tile_id", "num_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };
    if (sharded) {
        reader.compiler_options.defines = {{"SHARDED", "1"}};
        writer.compiler_options.defines = {{"SHARDED", "1"}};
    } else {
        reader.tensor_bindings = {m2::TensorBinding{T_A, "a"}, m2::TensorBinding{T_B, "b"}};
        writer.tensor_bindings = {m2::TensorBinding{T_OUT, "out"}};
    }

    m2::ProgramSpec spec{
        .name = "binary_practice_quasar",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {make_dfb(DFB_A, a, T_A), make_dfb(DFB_B, b, T_B), make_dfb(DFB_OUT, output, T_OUT)},
        .tensor_parameters =
            {{.unique_id = T_A, .spec = a.tensor_spec()},
             {.unique_id = T_B, .spec = b.tensor_spec()},
             {.unique_id = T_OUT, .spec = output.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{
            .name = "binary_practice_quasar",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = m2::NodeRangeSet(target_ranges)}},
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE, .runtime_arg_values = std::move(compute_args)},
    };
    run_params.tensor_args.emplace(T_A, m2::ProgramRunArgs::TensorArgument{a.mesh_tensor()});
    run_params.tensor_args.emplace(T_B, m2::ProgramRunArgs::TensorArgument{b.mesh_tensor()});
    run_params.tensor_args.emplace(T_OUT, m2::ProgramRunArgs::TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

}  // namespace ttnn::operations::experimental::binary_practice
