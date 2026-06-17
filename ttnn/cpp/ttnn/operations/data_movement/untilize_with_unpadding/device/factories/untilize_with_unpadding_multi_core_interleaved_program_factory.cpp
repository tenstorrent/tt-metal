// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"

#include <algorithm>
#include <vector>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

// Each BlockRep run is encoded as this many positional vararg words on the writer kernel.
// Layout (must match writer_unary_stick_layout_split_rows_multicore_metal2.cpp): {n_data, n_mixed,
// n_pads, times, repeat_count}.
static constexpr uint32_t kWordsPerBlockRepGroup = 5;

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_FULL_KERNEL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF_KERNEL{"compute_cliff"};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t num_blocks = input_shape[-1] == 0 ? 0 : a.physical_volume() / input_shape[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.ranges().empty();

    uint32_t padded_row_size_bytes;
    uint32_t unpadded_row_size_bytes;
    if (a.dtype() == DataType::BFLOAT8_B) {
        padded_row_size_bytes = input_shape[-1] * output.element_size();
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
    } else {
        padded_row_size_bytes = input_shape[-1] * a.element_size();
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
    }

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    bool float32_dtype = input_cb_data_format == tt::DataFormat::Float32 ||
                         input_cb_data_format == tt::DataFormat::UInt32 ||
                         input_cb_data_format == tt::DataFormat::Int32;

    // ------------------------------------------------------------------------
    // Dataflow buffers (formerly CB c_0 / c_16).
    // ------------------------------------------------------------------------
    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_mesh.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};

    // ------------------------------------------------------------------------
    // Per-core runtime args. This factory's variable-length descriptor stream lives on the WRITER:
    // its per-block BlockRep stream has a LENGTH that differs core-to-core. Metal 2.0 named RTAs
    // require a fixed schema (every node on a KernelSpec supplies the same named args), so the
    // variable stream cannot be a named RTA. Instead:
    //   - the writer's fixed scalars (padded_X_size, start_stick_id, n_block_reps) stay named RTAs;
    //   - the variable BlockRep stream is passed as positional "varargs"
    //     (KernelAdvancedOptions::num_runtime_varargs), read in the kernel via get_vararg().
    // Named-RTA validation requires every node to supply EXACTLY num_runtime_varargs words, so we
    // compute each core's stream, take the MAX length, declare that as the uniform vararg count, and
    // zero-pad shorter cores. n_block_reps bounds the kernel loop so the pad is never read. (We
    // deliberately avoid the deprecated per-node-vararg-count override.)
    // The reader is simple (one interleaved tile stream) and carries only per-core named RTAs.
    // ------------------------------------------------------------------------
    uint32_t tile_height = output.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    struct ReaderCoreArgs {
        NodeCoord node;
        uint32_t num_tiles;
        uint32_t start_page_id;
    };
    struct WriterCoreArgs {
        NodeCoord node;
        uint32_t start_stick_id;
        uint32_t n_block_reps;
        std::vector<uint32_t> block_rep_stream;  // run-length-encoded BlockRep groups (varargs)
    };

    std::vector<ReaderCoreArgs> reader_core_args;
    std::vector<WriterCoreArgs> writer_core_args;
    reader_core_args.reserve(ncores);
    writer_core_args.reserve(ncores);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    uint32_t max_stream_words = 0;

    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<ttnn::BlockRep>& assignment = core_assignments.at(i);
        const NodeCoord node = core;

        // start_stick_id is captured at the start of this core's work; the inner loop advances it.
        const uint32_t core_row_start_id = row_start_id;

        // Run-length-encode the assignment into {n_data, n_mixed, n_pads, times, repeat_count}
        // groups (identical to the legacy factory, but emitted into a vararg vector instead of a
        // flat RTA list).
        std::vector<uint32_t> stream;
        stream.reserve(assignment.size() * kWordsPerBlockRepGroup);
        uint32_t nblocks_per_core_core = 0;
        ttnn::BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;
        for (const auto& el : assignment) {
            nblocks_per_core_core += el.block_count();
            row_start_id += el.data_row_count();
            if (ttnn::compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                stream.push_back(ref_el.n_data);
                stream.push_back(ref_el.n_mixed);
                stream.push_back(ref_el.n_pads);
                stream.push_back(ref_el.times);
                stream.push_back(count_repeated);
                ref_el = el;
                count_repeated = 1;
            }
        }
        stream.push_back(ref_el.n_data);
        stream.push_back(ref_el.n_mixed);
        stream.push_back(ref_el.n_pads);
        stream.push_back(ref_el.times);
        stream.push_back(count_repeated);

        max_stream_words = std::max(max_stream_words, static_cast<uint32_t>(stream.size()));

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_core;
        reader_core_args.push_back(
            ReaderCoreArgs{.node = node, .num_tiles = num_tiles_per_core, .start_page_id = tile_start_id});
        writer_core_args.push_back(WriterCoreArgs{
            .node = node,
            .start_stick_id = core_row_start_id,
            .n_block_reps = static_cast<uint32_t>(assignment.size()),
            .block_rep_stream = std::move(stream),
        });

        tile_start_id += num_tiles_per_core;
    }

    // ------------------------------------------------------------------------
    // Kernels. Reader reuses untilize's ported single-tile interleaved reader; writer is the Metal
    // 2.0 fork carrying the vararg BlockRep stream; compute reuses untilize's ported compute fork.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp"},
        .dfb_bindings = {ProducerOf(IN_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_page_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_stick_layout_split_rows_multicore_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .compile_time_args = {{"float32_dtype", float32_dtype ? 1u : 0u}, {"unpadded_X_size", unpadded_row_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "start_stick_id", "n_block_reps"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
        // Variable-length BlockRep stream. Sized to the longest per-core stream; shorter cores are
        // zero-padded below. See the per-core args comment above. (advanced_options is the last
        // KernelSpec field, so it must follow hw_config in this designated initializer.)
        .advanced_options = KernelAdvancedOptions{.num_runtime_varargs = max_stream_words},
    };

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_dest_acc_en) {
        unpack_to_dest_modes.insert({IN_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    // The legacy factory created a dedicated cliff compute kernel with a different compile-time
    // block count. That maps 1:1 to two compute KernelSpecs of the same source (full + cliff), each
    // in its own WorkUnitSpec, both binding the shared in/out DFBs. per_core_block_cnt stays a
    // compile-time arg (preserves loop unrolling).
    auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks_this_group) {
        return KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
                                            "untilize_compute_metal2.cpp"},
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings = {ConsumerOf(IN_DFB, "in"), ProducerOf(OUT_DFB, "out")},
            .compile_time_args =
                {{"per_core_block_cnt", nblocks_this_group}, {"per_core_block_tile_cnt", num_tiles_per_row}},
            .hw_config =
                ComputeHardwareConfig{
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .unpack_to_dest_mode = unpack_to_dest_modes,
                },
        };
    };

    std::vector<KernelSpec> kernels = {reader_spec, writer_spec, make_compute(COMPUTE_FULL_KERNEL, nblocks_per_core)};
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF_KERNEL, nblocks_per_core_cliff));
    }

    // ------------------------------------------------------------------------
    // Per-node runtime args. Reader: named scalars only. Writer: named scalars + the BlockRep
    // stream via advanced_options.runtime_varargs (zero-padded to max_stream_words).
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.runtime_arg_values.reserve(ncores);
    writer_run.runtime_arg_values.reserve(ncores);

    for (const auto& rca : reader_core_args) {
        reader_run.runtime_arg_values.push_back(
            {rca.node, {{"num_tiles", rca.num_tiles}, {"start_page_id", rca.start_page_id}}});
    }
    for (auto& wca : writer_core_args) {
        writer_run.runtime_arg_values.push_back(
            {wca.node,
             {{"padded_X_size", padded_row_size_bytes},
              {"start_stick_id", wca.start_stick_id},
              {"n_block_reps", wca.n_block_reps}}});

        std::vector<uint32_t> padded_stream = std::move(wca.block_rep_stream);
        padded_stream.resize(max_stream_words, 0u);  // uniform vararg count across all cores
        // runtime_varargs is a Table<NodeCoord, Varargs> (no push_back); insert one entry per node.
        writer_run.advanced_options.runtime_varargs[wca.node] = std::move(padded_stream);
    }

    std::vector<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "untilize_with_unpadding_multi_core_interleaved_full",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_FULL_KERNEL},
        .target_nodes = core_range,
    });
    if (has_cliff) {
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_with_unpadding_multi_core_interleaved_cliff",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_KERNEL},
            .target_nodes = core_range_cliff,
        });
    }

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_mesh)}}, {OUTPUT_TENSOR, TensorArgument{std::cref(output_mesh)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
