// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using tt::tt_metal::experimental::AdvancedKernelRunArgs;

namespace ttnn::prim {

namespace {

// Spec names are prefixed per factory: all five factory .cpp files land in one unity-build
// translation unit, where every anonymous namespace merges into a single scope.
const KernelSpecName MCI_READER{"mci_reader"};
const KernelSpecName MCI_WRITER{"mci_writer"};
const KernelSpecName MCI_COMPUTE{"mci_compute"};
const DFBSpecName MCI_IN{"mci_in"};
const DFBSpecName MCI_OUT{"mci_out"};
const TensorParamName MCI_INPUT{"mci_input"};
const TensorParamName MCI_OUTPUT{"mci_output"};

// A shape-free program gives every core of the grid this many writer varargs, five words per run of equal
// BlockReps. A core's blocks are one contiguous range of the row pattern, which is a repeat of one BlockRep:
// the shape-free check admits padding in the last two dims only, and at most 4D shapes (read as 4D, padded
// with ones). So a core has at most two pieces of a split rep at either end, one run of whole reps between
// them and the empty trailing pad entry, six runs; the cap leaves two runs of margin.
constexpr uint32_t MCI_MAX_WRITER_RUNS = 8;
constexpr uint32_t MCI_WORDS_PER_RUN = 5;

uint32_t mci_num_blocks(const Tensor& input) {
    const auto& input_shape = input.padded_shape();
    return input_shape[-1] == 0 ? 0 : input.physical_volume() / input_shape[-1] / TILE_HEIGHT;
}

CoreRangeSet mci_available_grid(const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input) {
    if (operation_attributes.sub_core_grids.has_value()) {
        return operation_attributes.sub_core_grids.value();
    }
    const CoreCoord grid_size = input.device()->compute_with_storage_grid_size();
    return CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));
}

struct MciRunArgs {
    ProgramRunArgs run_args;
    // The writer's vararg count per core, which the spec declares.
    std::map<CoreCoord, uint32_t> writer_varargs_per_node;
};

// Every runtime arg of the program, which follows the shapes: each core's tile range, block count, first stick
// and BlockRep runs. A shape-free program (untilize_with_unpadding_interleaved_is_shape_free) covers the whole
// available grid, so cores past the split get no work, and gives every core the same vararg count.
MciRunArgs mci_run_args(
    const UntilizeWithUnpaddingParams& operation_attributes,
    const Tensor& input,
    const Tensor& output,
    bool shape_free) {
    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    const CoreRangeSet available_grid = mci_available_grid(operation_attributes, input);
    const uint32_t num_blocks = mci_num_blocks(input);
    const uint32_t num_tiles_per_row = input_shape[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);
    const bool has_cliff = !core_range_cliff.empty();

    const uint32_t element_size = input.dtype() == DataType::BFLOAT8_B ? output.element_size() : input.element_size();
    const uint32_t padded_row_size_bytes = input_shape[-1] * element_size;

    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    MciRunArgs result;
    KernelRunArgs reader_run_args{.kernel = MCI_READER};
    KernelRunArgs writer_run_args{.kernel = MCI_WRITER};
    KernelRunArgs compute_run_args{.kernel = MCI_COMPUTE};
    // Each arg's per-core table, looked up once: a cache hit at another shape rebuilds them all.
    auto& per_core_block_cnt = compute_run_args.runtime_arg_values["per_core_block_cnt"];
    auto& num_pages = reader_run_args.runtime_arg_values["num_pages"];
    auto& start_id = reader_run_args.runtime_arg_values["start_id"];
    auto& padded_X_size = writer_run_args.runtime_arg_values["padded_X_size"];
    auto& start_stick_id = writer_run_args.runtime_arg_values["start_stick_id"];
    auto& n_block_reps_arg = writer_run_args.runtime_arg_values["n_block_reps"];
    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);
    const uint32_t num_program_cores = shape_free ? static_cast<uint32_t>(cores.size()) : ncores;
    for (uint32_t i = 0; i < num_program_cores; ++i) {
        const auto& core = cores[i];
        AdvancedKernelRunArgs::Varargs writer_varargs;
        const uint32_t core_row_start_id = row_start_id;
        uint32_t nblocks_per_core_core = 0;
        uint32_t n_block_reps = 0;
        if (i < ncores) {
            const std::vector<BlockRep>& assignment = core_assignments.at(i);
            n_block_reps = static_cast<uint32_t>(assignment.size());
            BlockRep ref_el = assignment[0];
            uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
            auto push_run = [&](const BlockRep& el, uint32_t repeated) {
                writer_varargs.push_back(el.n_data);
                writer_varargs.push_back(el.n_mixed);
                writer_varargs.push_back(el.n_pads);
                writer_varargs.push_back(el.times);
                writer_varargs.push_back(repeated);
            };
            for (const auto& el : assignment) {
                nblocks_per_core_core += el.block_count();
                row_start_id += el.data_row_count();
                if (compare_assignments(ref_el, el)) {
                    count_repeated++;
                } else {
                    push_run(ref_el, count_repeated);
                    ref_el = el;
                    count_repeated = 1;
                }
            }
            push_run(ref_el, count_repeated);
        }
        if (shape_free) {
            TT_FATAL(
                writer_varargs.size() <= MCI_MAX_WRITER_RUNS * MCI_WORDS_PER_RUN,
                "untilize_with_unpadding: core {} needs {} BlockRep runs, more than the {} a shape-free program has",
                i,
                writer_varargs.size() / MCI_WORDS_PER_RUN,
                MCI_MAX_WRITER_RUNS);
            writer_varargs.resize(MCI_MAX_WRITER_RUNS * MCI_WORDS_PER_RUN, 0);
        }

        const uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_core;
        per_core_block_cnt[core] = nblocks_per_core_core;
        num_pages[core] = num_tiles_per_core;
        start_id[core] = tile_start_id;
        padded_X_size[core] = padded_row_size_bytes;
        start_stick_id[core] = core_row_start_id;
        n_block_reps_arg[core] = n_block_reps;
        // Each core's BlockRep runs are a runtime-length payload, so they ride as runtime varargs.
        result.writer_varargs_per_node[core] = static_cast<uint32_t>(writer_varargs.size());
        writer_run_args.advanced_options.runtime_varargs[core] = std::move(writer_varargs);

        tile_start_id += num_tiles_per_core;
    }

    result.run_args.kernel_run_args = {
        std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    result.run_args.tensor_args = {
        {MCI_INPUT, input.mesh_tensor()},
        {MCI_OUTPUT, output.mesh_tensor()},
    };
    return result;
}

}  // namespace

bool untilize_with_unpadding_interleaved_is_shape_free(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input) {
    // The kernels take every count at run time and the buffers follow the row width alone, and the program spans
    // the whole available grid, so it serves any shape with the same last dim. That costs host time on every hit
    // (all cores' runtime args are rebuilt) and idle cores below one block per core, hence the opt-in.
    // The writer's vararg count (MCI_MAX_WRITER_RUNS) holds only while the unpadding is in the last two dims.
    const auto& padded_shape = input.padded_shape();
    const auto& end = operation_attributes.output_tensor_end;
    if (!operation_attributes.runtime_shapes || input.is_sharded() ||
        operation_attributes.output_mem_config.is_sharded() || padded_shape.rank() > 4 ||
        end.rank() != padded_shape.rank()) {
        return false;
    }
    for (int i = 0; i + 2 < static_cast<int>(padded_shape.rank()); ++i) {
        if (end[i] + 1 != padded_shape[i]) {
            return false;
        }
    }
    return true;
}

ProgramRunArgs UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::override_runtime_arguments(
    const UntilizeWithUnpaddingParams& operation_attributes,
    const Tensor& input,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    if (!untilize_with_unpadding_interleaved_is_shape_free(operation_attributes, input)) {
        // The key holds the full shapes, so only the tensors change.
        ProgramRunArgs run_args;
        run_args.tensor_args = {
            {MCI_INPUT, input.mesh_tensor()},
            {MCI_OUTPUT, output.mesh_tensor()},
        };
        return run_args;
    }
    return mci_run_args(operation_attributes, input, output, /*shape_free=*/true).run_args;
}

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat input_dfb_data_format = get_untilize_with_unpadding_dfb_data_format(a.dtype(), a.device()->arch());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);
    tt::DataFormat output_dfb_data_format =
        get_untilize_with_unpadding_dfb_data_format(output.dtype(), a.device()->arch());
    uint32_t output_single_tile_size = tt::tile_size(output_dfb_data_format);

    const auto& output_shape = output.padded_shape();
    const bool shape_free = untilize_with_unpadding_interleaved_is_shape_free(operation_attributes, input);
    const CoreRangeSet available_grid = mci_available_grid(operation_attributes, input);
    const uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, mci_num_blocks(input));

    const uint32_t element_size = a.dtype() == DataType::BFLOAT8_B ? output.element_size() : a.element_size();
    const uint32_t unpadded_row_size_bytes = output_shape[-1] * element_size;

    TT_ASSERT(output.is_allocated(), "Output buffer should be allocated on device!");

    DataflowBufferSpec in_dfb{
        .unique_id = MCI_IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = input_dfb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = MCI_OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = output_dfb_data_format,
    };

    /** reader
     */
    KernelSpec reader{
        .unique_id = MCI_READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = MCI_IN,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = MCI_INPUT,
            .accessor_name = "src",
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        // The DM kernels credit the DFBs explicitly. A whole-row ring (one entry per tile of the padded width)
        // exceeds the 8-bit implicit-sync descriptor past 255 tiles and a prime width (e.g. 263) can never
        // be split across transaction ids, so implicit sync is switched off here as in tilize_with_val_padding.
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    /** writer
     */
    // The destination's per-shard page size — the shard width for BLOCK/WIDTH-sharded output (a
    // logical row spans multiple shards), the buffer's aligned page size for HEIGHT-sharded output
    // (one row per page, padded up to the buffer alignment), the full unpadded row for interleaved —
    // is supplied by the tensor binding, which hands the kernel the tensor's aligned page size.
    KernelSpec writer{
        .unique_id = MCI_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_split_rows_multicore.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = MCI_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = MCI_OUTPUT,
            .accessor_name = "dst",
        }},
        .compile_time_args =
            {{"float32_dtype",
              (uint32_t)(input_dfb_data_format == tt::DataFormat::Float32 or
                         input_dfb_data_format == tt::DataFormat::UInt32 or
                         input_dfb_data_format == tt::DataFormat::Int32)},
             {"one_byte_dtype",
              (uint32_t)(input_dfb_data_format == tt::DataFormat::UInt8 ||
                         input_dfb_data_format == tt::DataFormat::Int8)},
             {"unpadded_X_size", unpadded_row_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "start_stick_id", "n_block_reps"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    /** compute
     */
    ComputeHardwareConfig compute_hw_config{.enable_32_bit_dest = fp32_dest_acc_en};
    if (operations::data_movement::needs_fp32_unpack_to_dest(fp32_dest_acc_en, a.dtype())) {
        compute_hw_config.unpack_modes = {{MCI_IN, UnpackMode::UnpackToDest}};
    }

    // One compute kernel for the full and the cliff cores: the block count per core is a runtime
    // arg, so the groups differ only in their runtime args and the binary does not depend on the
    // height.
    Group<KernelSpec> kernels = {
        std::move(reader),
        KernelSpec{
            .unique_id = MCI_COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp",
            .compiler_options =
                {
                    .defines = {{"UNTILIZE_RUNTIME_BLOCK_CNT", "1"}},
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = MCI_IN,
                     .accessor_name = "src",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = MCI_OUT,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args = {{"per_core_block_tile_cnt", num_tiles_per_row}},
            .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt"}},
            .hw_config = compute_hw_config,
        }};
    // A shape-free program runs on the whole available grid, the cores past the split with no work, so its
    // core set does not follow the block count. One work unit serves both groups: the block count per core is a
    // runtime arg.
    Group<WorkUnitSpec> work_units;
    if (shape_free) {
        work_units.push_back(WorkUnitSpec{
            .name = "all",
            .kernels = {MCI_READER, MCI_WRITER, MCI_COMPUTE},
            .target_nodes = available_grid,
        });
    } else {
        if (!core_range.empty()) {
            work_units.push_back(WorkUnitSpec{
                .name = "full",
                .kernels = {MCI_READER, MCI_WRITER, MCI_COMPUTE},
                .target_nodes = core_range,
            });
        }
        if (!core_range_cliff.empty()) {
            work_units.push_back(WorkUnitSpec{
                .name = "cliff",
                .kernels = {MCI_READER, MCI_WRITER, MCI_COMPUTE},
                .target_nodes = core_range_cliff,
            });
        }
    }

    MciRunArgs run_args = mci_run_args(operation_attributes, input, output, shape_free);
    // A shape-free program gives every core the same vararg count. Otherwise the run count differs per core,
    // which is what the per-node vararg-count override says.
    if (shape_free) {
        writer.advanced_options.num_runtime_varargs = MCI_MAX_WRITER_RUNS * MCI_WORDS_PER_RUN;
    } else {
        for (const auto& [core, num_varargs] : run_args.writer_varargs_per_node) {
            writer.advanced_options.num_runtime_varargs_per_node[core] = num_varargs;
        }
    }
    kernels.push_back(std::move(writer));

    // A shape-free program serves other shapes with the same last dim, so its tensors may differ in shape.
    TensorSpecRelaxations relaxations;
    relaxations.dynamic_tensor_shape = shape_free;
    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = {std::move(in_dfb), std::move(out_dfb)},
        .tensor_parameters =
            {TensorParameter{
                 .unique_id = MCI_INPUT, .spec = input_mesh_tensor.tensor_spec(), .relaxations = relaxations},
             TensorParameter{
                 .unique_id = MCI_OUTPUT, .spec = output_mesh_tensor.tensor_spec(), .relaxations = relaxations}},
        .work_units = std::move(work_units),
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args.run_args),
    };
}

}  // namespace ttnn::prim
